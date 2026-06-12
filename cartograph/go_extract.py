"""tree-sitter extraction for Go -> the same graph model.

Optional `go` extra (tree-sitter-go, MIT). Produces module/struct(class)/interface/
function/method nodes with CONTAINS, INHERITS (struct embedding — deterministic
composition, the closest Go has), IMPORTS, and heuristic INFERRED CALLS. Receiver
methods (`func (o *Order) IsPaid()`) attach to their receiver type, and calls
through the receiver variable (`o.check()`) disambiguate to that type's methods.
Implicit interface satisfaction needs a type checker and is NOT extracted. No network.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .model import EXTRACTED, INFERRED, Edge, Graph, Node


def _parser():
    import tree_sitter_go as tsg
    from tree_sitter import Language, Parser

    return Parser(Language(tsg.language()))


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8", "replace")).hexdigest()


class _GoFile:
    def __init__(self, src: bytes, rel_path: str):
        self.src = src
        self.rel_path = rel_path
        self.package = ""
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.calls: list[tuple[str, str, bool]] = []   # (caller_id, name, is_receiver)
        self.bases: list[tuple[str, str]] = []          # (struct_id, embedded_type)
        self.imports: list[tuple[str, str]] = []        # (module_id, path)
        self.module_node: Node | None = None
        self.types: dict[str, Node] = {}                # local type name -> node
        # receiver-method ownership is resolved in the cross-file join phase:
        # the receiver type often lives in another file of the same package
        self.pending_owns: list[tuple[str, str]] = []   # (method_id, owner_qual)

    def _text(self, n) -> str:
        return self.src[n.start_byte:n.end_byte].decode("utf-8", "replace")

    def _first_line(self, n) -> str:
        return re.sub(r"\s+", " ", self._text(n).split("\n", 1)[0]).strip().rstrip("{").strip()

    def _node(self, kind: str, name: str, qual: str, n) -> Node:
        line = n.start_point[0] + 1
        node = Node(
            id=f"{self.rel_path}::{qual}#{line}", kind=kind, name=name, qualified_name=qual,
            module=self.module_node.qualified_name if self.module_node else qual,
            file_path=self.rel_path, start_line=line, end_line=n.end_point[0] + 1,
            signature=self._first_line(n),
            embed_text=f"{kind} {qual}\n{self._first_line(n)}".strip(),
            content_sha=_sha(self._text(n)),
        )
        self.nodes.append(node)
        return node

    def run(self, root) -> None:
        for child in root.children:
            if child.type == "package_clause":
                ids = [c for c in child.named_children if c.type == "package_identifier"]
                if ids:
                    self.package = self._text(ids[0])
                break
        stem = Path(self.rel_path).stem
        mod_qual = f"{self.package}.{stem}" if self.package else stem
        self.module_node = Node(
            id=f"{self.rel_path}::{mod_qual}", kind="module", name=stem,
            qualified_name=mod_qual, module=mod_qual, file_path=self.rel_path,
            start_line=1, end_line=root.end_point[0] + 1,
            embed_text=f"module {mod_qual}", content_sha=_sha(self._text(root)),
        )
        self.nodes.append(self.module_node)
        scope = self.package or stem
        for child in root.children:
            if child.type == "import_declaration":
                for spec in child.named_children:
                    specs = spec.named_children if spec.type == "import_spec_list" else [spec]
                    for s in specs:
                        if s.type == "import_spec":
                            lit = next((c for c in s.named_children
                                        if c.type == "interpreted_string_literal"), None)
                            if lit is not None:
                                self.imports.append((self.module_node.id, self._text(lit).strip('"')))
            elif child.type == "type_declaration":
                for spec in (c for c in child.named_children if c.type == "type_spec"):
                    self._type_spec(spec, scope)
            elif child.type == "function_declaration":
                name_n = child.child_by_field_name("name")
                if name_n is None:
                    continue
                name = self._text(name_n)
                fn = self._node("function", name, f"{scope}.{name}", child)
                self.edges.append(Edge("CONTAINS", self.module_node.id, fn.id, EXTRACTED))
                body = child.child_by_field_name("body")
                if body is not None:
                    self._collect_calls(body, fn.id, receiver_var=None)
            elif child.type == "method_declaration":
                self._method(child, scope)

    def _type_spec(self, spec, scope: str) -> None:
        name_n = spec.child_by_field_name("name")
        type_n = spec.child_by_field_name("type")
        if name_n is None or type_n is None:
            return
        name = self._text(name_n)
        kind = "interface" if type_n.type == "interface_type" else "class"
        node = self._node(kind, name, f"{scope}.{name}", spec)
        self.types[name] = node
        self.edges.append(Edge("CONTAINS", self.module_node.id, node.id, EXTRACTED))
        if type_n.type == "struct_type":  # embedded types: deterministic composition
            fields = next((c for c in type_n.named_children
                           if c.type == "field_declaration_list"), None)
            for fd in (fields.named_children if fields is not None else ()):
                if fd.type != "field_declaration":
                    continue
                named = fd.named_children
                # embedding = a field with a type but no field name
                if len(named) == 1 and named[0].type in ("type_identifier", "qualified_type", "pointer_type"):
                    self.bases.append((node.id, self._text(named[0]).lstrip("*").split(".")[-1]))

    def _method(self, decl, scope: str) -> None:
        recv = next((c for c in decl.children if c.type == "parameter_list"), None)
        name_n = decl.child_by_field_name("name")
        if name_n is None:
            return
        recv_type, recv_var = None, None
        if recv is not None:
            pd = next((c for c in recv.named_children if c.type == "parameter_declaration"), None)
            if pd is not None:
                tn = pd.child_by_field_name("type")
                if tn is not None:
                    recv_type = self._text(tn).lstrip("*").split(".")[-1].split("[")[0]
                vn = pd.child_by_field_name("name")
                if vn is not None:
                    recv_var = self._text(vn)
        name = self._text(name_n)
        qual = f"{scope}.{recv_type}.{name}" if recv_type else f"{scope}.{name}"
        meth = self._node("method" if recv_type else "function", name, qual, decl)
        if recv_type:
            self.pending_owns.append((meth.id, f"{scope}.{recv_type}"))
        else:
            self.edges.append(Edge("CONTAINS", self.module_node.id, meth.id, EXTRACTED))
        body = decl.child_by_field_name("body")
        if body is not None:
            self._collect_calls(body, meth.id, receiver_var=recv_var)

    def _collect_calls(self, node, caller_id: str, receiver_var: str | None) -> None:
        if node.type == "call_expression":
            fn = node.child_by_field_name("function")
            if fn is not None:
                if fn.type == "identifier":
                    self.calls.append((caller_id, self._text(fn), False))
                elif fn.type == "selector_expression":
                    operand = fn.child_by_field_name("operand")
                    field = fn.child_by_field_name("field")
                    if field is not None:
                        is_recv = (operand is not None and operand.type == "identifier"
                                   and receiver_var is not None
                                   and self._text(operand) == receiver_var)
                        self.calls.append((caller_id, self._text(field), is_recv))
        for child in node.children:
            if child.type in ("function_declaration", "method_declaration"):
                continue
            self._collect_calls(child, caller_id, receiver_var)


def extract_go_paths(paths: list[Path], root: Path) -> Graph:
    parser = _parser()
    pkg_parent = root.parent
    files: list[_GoFile] = []
    for path in paths:
        rel = path.relative_to(pkg_parent).as_posix() if path.is_relative_to(pkg_parent) else path.name
        src = path.read_bytes()
        fx = _GoFile(src, rel)
        fx.run(parser.parse(src).root_node)
        files.append(fx)

    nodes: list[Node] = [n for f in files for n in f.nodes]
    name_index: dict[str, list[Node]] = {}
    for n in nodes:
        if n.kind in ("function", "method", "class", "interface"):
            name_index.setdefault(n.name, []).append(n)
    by_id = {n.id: n for n in nodes}
    edges: list[Edge] = list({(e.type, e.src, e.dst): e for f in files for e in f.edges}.values())
    seen = {(e.type, e.src, e.dst) for e in edges}

    # Attach receiver methods to their type across files of the same package.
    type_index = {n.qualified_name: n for n in nodes if n.kind in ("class", "interface")}
    for f in files:
        for meth_id, owner_qual in f.pending_owns:
            owner = type_index.get(owner_qual)
            parent = owner.id if owner is not None else f.module_node.id
            if ("CONTAINS", parent, meth_id) not in seen:
                seen.add(("CONTAINS", parent, meth_id))
                edges.append(Edge("CONTAINS", parent, meth_id, EXTRACTED))

    def _type_of(n: Node) -> str | None:
        return n.qualified_name.rsplit(".", 1)[0] if n.kind == "method" else None

    for f in files:
        for caller_id, name, is_recv in f.calls:
            caller = by_id.get(caller_id)
            cands = name_index.get(name, [])
            chosen = None
            if is_recv and caller is not None and caller.kind == "method":
                same = [c for c in cands if _type_of(c) == _type_of(caller)]
                if same:
                    chosen = same
            if chosen is None:
                same_mod = [c for c in cands if caller and c.module == caller.module]
                chosen = same_mod or cands
            if len(chosen) > 8:
                continue
            for c in chosen:
                if c.id != caller_id and ("CALLS", caller_id, c.id) not in seen:
                    seen.add(("CALLS", caller_id, c.id))
                    edges.append(Edge("CALLS", caller_id, c.id, INFERRED, "tree-sitter-go"))

    for f in files:
        for struct_id, base in f.bases:
            matches = [c for c in name_index.get(base, [])
                       if c.kind in ("class", "interface") and c.id != struct_id]
            confidence = EXTRACTED if len(matches) == 1 else INFERRED
            for c in matches:
                if ("INHERITS", struct_id, c.id) not in seen:
                    seen.add(("INHERITS", struct_id, c.id))
                    edges.append(Edge("INHERITS", struct_id, c.id, confidence))

    ext: dict[str, Node] = {}
    for f in files:
        for mod_id, target in f.imports:
            e = ext.get(target)
            if e is None:
                e = Node(id=f"ext::{target}", kind="external", name=target.rsplit("/", 1)[-1],
                         qualified_name=target, module=target, file_path="<external>",
                         start_line=0, end_line=0, embed_text=f"import {target}")
                ext[target] = e
            if ("IMPORTS", mod_id, e.id) not in seen:
                seen.add(("IMPORTS", mod_id, e.id))
                edges.append(Edge("IMPORTS", mod_id, e.id, EXTRACTED))
    nodes.extend(ext.values())
    return Graph(nodes=nodes, edges=edges)
