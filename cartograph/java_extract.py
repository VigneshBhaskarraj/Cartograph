"""tree-sitter extraction for Java -> the same graph model.

Optional `java` extra (tree-sitter-java, MIT). Produces module/class/interface/
enum/method nodes with CONTAINS, INHERITS (extends/implements), IMPORTS, and
heuristic INFERRED CALLS — mirroring the Python/TS extractors. The JPA annotations
@Entity/@Table/@Column feed the code<->data bridge (MAPS_TO), which is where Java
matters most: enterprise codebases declare their schema mapping explicitly.
Lexical scope is threaded through the walk (nested/inner classes carry true
qualified names — the lesson from the Python extractor audit). No network.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .model import EXTRACTED, INFERRED, Edge, Graph, Node

_TYPE_DECLS = ("class_declaration", "interface_declaration", "enum_declaration",
               "record_declaration", "annotation_type_declaration")
_KIND = {"class_declaration": "class", "interface_declaration": "interface",
         "enum_declaration": "class", "record_declaration": "class",
         "annotation_type_declaration": "interface"}


def _parser():
    import tree_sitter_java as tsj
    from tree_sitter import Language, Parser

    return Parser(Language(tsj.language()))


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8", "replace")).hexdigest()


def _javadoc(text: str) -> str:
    """Plain prose from a `/** … */` javadoc block; drops the `*` gutter and
    @param/@return/@throws tag lines (the natural-language summary is what carries
    SEMANTIC recall, mirroring how Python docstrings feed embed_text)."""
    t = text.strip()
    if not t.startswith("/**"):
        return ""  # only javadoc, not ordinary /* */ block comments
    t = t[3:-2] if t.endswith("*/") else t[3:]
    out = []
    for ln in t.splitlines():
        ln = ln.strip().lstrip("*").strip()
        if ln and not ln.startswith("@"):
            out.append(ln)
    return " ".join(out)


def _preceding_doc(decl) -> str:
    """Javadoc immediately preceding a declaration (its prev sibling)."""
    sib = decl.prev_sibling
    if sib is not None and sib.type in ("block_comment", "comment"):
        return _javadoc(sib.text.decode("utf-8", "replace"))
    return ""


class _JavaFile:
    def __init__(self, src: bytes, rel_path: str):
        self.src = src
        self.rel_path = rel_path
        self.package = ""
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.calls: list[tuple[str, str, bool]] = []   # (caller_id, name, is_this)
        self.bases: list[tuple[str, str]] = []          # (type_id, base_name)
        self.imports: list[tuple[str, str]] = []        # (module_id, target)
        self.module_node: Node | None = None

    def _text(self, n) -> str:
        return self.src[n.start_byte:n.end_byte].decode("utf-8", "replace")

    def _first_line(self, n) -> str:
        return re.sub(r"\s+", " ", self._text(n).split("\n", 1)[0]).strip().rstrip("{").strip()

    def _node(self, kind: str, name: str, qual: str, n, doc: str = "") -> Node:
        line = n.start_point[0] + 1
        embed = f"{kind} {qual}\n{self._first_line(n)}"
        if doc:
            embed += f"\n{doc}"  # javadoc prose drives SEMANTIC recall
        node = Node(
            id=f"{self.rel_path}::{qual}#{line}", kind=kind, name=name, qualified_name=qual,
            module=self.module_node.qualified_name if self.module_node else qual,
            file_path=self.rel_path, start_line=line, end_line=n.end_point[0] + 1,
            signature=self._first_line(n), docstring=doc,
            embed_text=embed.strip(),
            content_sha=_sha(self._text(n)),
        )
        self.nodes.append(node)
        return node

    # -- annotations ------------------------------------------------------------
    def _annotations(self, decl) -> dict[str, dict[str, str]]:
        """{annotation_name: {arg_key: value}} from a declaration's modifiers."""
        out: dict[str, dict[str, str]] = {}
        mods = next((c for c in decl.children if c.type == "modifiers"), None)
        if mods is None:
            return out
        for a in mods.children:
            if a.type not in ("annotation", "marker_annotation"):
                continue
            name_n = a.child_by_field_name("name")
            if name_n is None:
                continue
            args: dict[str, str] = {}
            arglist = a.child_by_field_name("arguments")
            if arglist is not None:
                for pair in arglist.named_children:
                    if pair.type == "element_value_pair":
                        k = pair.child_by_field_name("key")
                        v = pair.child_by_field_name("value")
                        if k is not None and v is not None:
                            args[self._text(k)] = self._text(v).strip('"')
                    elif pair.type == "string_literal":  # @Table("orders") single-value
                        args["value"] = self._text(pair).strip('"')
            out[self._text(name_n)] = args
        return out

    # -- traversal ----------------------------------------------------------------
    def run(self, root) -> None:
        for child in root.children:
            if child.type == "package_declaration":
                ids = [c for c in child.named_children if c.type in ("scoped_identifier", "identifier")]
                if ids:
                    self.package = self._text(ids[0])
                break
        stem = Path(self.rel_path).stem
        mod_qual = f"{self.package}.{stem}" if self.package else stem
        self.module_node = Node(
            id=f"{self.rel_path}::{mod_qual}", kind="module", name=stem,
            qualified_name=mod_qual, module=mod_qual, file_path=self.rel_path,
            start_line=1, end_line=root.end_point[0] + 1,
            embed_text=f"module {mod_qual}",
            content_sha=_sha(self._text(root)),
        )
        self.nodes.append(self.module_node)
        scope = self.package or stem
        for child in root.children:
            if child.type == "import_declaration":
                ids = [c for c in child.named_children if c.type in ("scoped_identifier", "identifier")]
                if ids:
                    self.imports.append((self.module_node.id, self._text(ids[0])))
            elif child.type in _TYPE_DECLS:
                self._type_decl(child, self.module_node.id, scope)

    def _type_decl(self, decl, parent_id: str, enclosing_qual: str) -> None:
        name_n = decl.child_by_field_name("name")
        if name_n is None:
            return
        name = self._text(name_n)
        qual = f"{enclosing_qual}.{name}"
        node = self._node(_KIND[decl.type], name, qual, decl, doc=_preceding_doc(decl))
        self.edges.append(Edge("CONTAINS", parent_id, node.id, EXTRACTED))
        annos = self._annotations(decl)
        if "Entity" in annos or "Table" in annos:
            explicit = annos.get("Table", {}).get("name") or annos.get("Table", {}).get("value")
            if explicit:
                node.extra["tablename"] = explicit
                node.extra["tablename_confidence"] = EXTRACTED  # declared mapping
            else:
                # bare @Entity: the table name depends on the naming strategy
                # (Spring Boot snake_cases) — a guess, so honestly INFERRED
                node.extra["tablename"] = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()
                node.extra["tablename_confidence"] = INFERRED
            node.extra["columns"] = []
        sup = next((c for c in decl.children if c.type == "superclass"), None)
        if sup is not None:
            for t in sup.named_children:
                if t.type in ("type_identifier", "generic_type"):
                    self.bases.append((node.id, self._text(t).split("<")[0]))
        ifaces = next((c for c in decl.children if c.type == "super_interfaces"), None)
        if ifaces is not None:
            for t in (d for lst in ifaces.named_children for d in lst.named_children):
                if t.type in ("type_identifier", "generic_type"):
                    self.bases.append((node.id, self._text(t).split("<")[0]))
        body = decl.child_by_field_name("body")
        if body is not None:
            self._type_body(body, node, qual)

    def _type_body(self, body, owner: Node, enclosing_qual: str) -> None:
        for m in body.named_children:
            if m.type in ("method_declaration", "constructor_declaration"):
                name_n = m.child_by_field_name("name")
                if name_n is None:
                    continue
                name = self._text(name_n)
                meth = self._node("method", name, f"{enclosing_qual}.{name}", m, doc=_preceding_doc(m))
                self.edges.append(Edge("CONTAINS", owner.id, meth.id, EXTRACTED))
                blk = m.child_by_field_name("body")
                if blk is not None:
                    self._collect_calls(blk, meth.id)
            elif m.type == "field_declaration":
                if "columns" in owner.extra:
                    annos = self._annotations(m)
                    decl = next((c for c in m.named_children if c.type == "variable_declarator"), None)
                    field = self._text(decl.child_by_field_name("name")) if decl is not None else None
                    if "Column" in annos or "JoinColumn" in annos:
                        col = (annos.get("Column", annos.get("JoinColumn", {})).get("name")
                               or annos.get("Column", annos.get("JoinColumn", {})).get("value")
                               or field)
                        if col:
                            owner.extra["columns"].append(col)
            elif m.type == "enum_body_declarations":  # enum methods live one level down
                self._type_body(m, owner, enclosing_qual)
            elif m.type in _TYPE_DECLS:  # nested/inner types carry true scope
                self._type_decl(m, owner.id, enclosing_qual)

    def _collect_calls(self, node, caller_id: str) -> None:
        if node.type == "method_invocation":
            name_n = node.child_by_field_name("name")
            obj = node.child_by_field_name("object")
            if name_n is not None:
                is_this = obj is None or (obj.type == "this")
                self.calls.append((caller_id, self._text(name_n), is_this))
        elif node.type == "object_creation_expression":
            t = node.child_by_field_name("type")
            if t is not None:
                self.calls.append((caller_id, self._text(t).split("<")[0], False))
        for child in node.children:
            if child.type in _TYPE_DECLS:
                continue  # nested types own their own calls
            self._collect_calls(child, caller_id)


def extract_java_paths(paths: list[Path], root: Path) -> Graph:
    parser = _parser()
    pkg_parent = root.parent
    files: list[_JavaFile] = []
    for path in paths:
        rel = path.relative_to(pkg_parent).as_posix() if path.is_relative_to(pkg_parent) else path.name
        src = path.read_bytes()
        fx = _JavaFile(src, rel)
        tree = parser.parse(src)
        if tree.root_node.has_error:
            import warnings  # tree-sitter never raises; partial graphs must not be silent (G5-C6)
            warnings.warn(f"syntax errors in {rel}; its graph may be partial", stacklevel=2)
        fx.run(tree.root_node)
        files.append(fx)

    nodes: list[Node] = [n for f in files for n in f.nodes]
    name_index: dict[str, list[Node]] = {}
    qual_index: dict[str, Node] = {}
    for n in nodes:
        if n.kind in ("method", "class", "interface"):
            name_index.setdefault(n.name, []).append(n)
        if n.kind in ("class", "interface"):
            qual_index.setdefault(n.qualified_name, n)
    by_id = {n.id: n for n in nodes}
    # Java resolves unqualified names class-then-package-wide, but node.module is
    # per-FILE ("<pkg>.<stem>") — comparing modules made the "same scope" tier
    # mean "same file" and let cross-package name collisions through (G5-C3).
    pkg_of = {n.id: f.package for f in files for n in f.nodes}
    edges: list[Edge] = list({(e.type, e.src, e.dst): e for f in files for e in f.edges}.values())
    seen = {(e.type, e.src, e.dst) for e in edges}

    # Shared tiered resolution (G5-C5): Java's lexical unit is the PACKAGE.
    from .resolve import resolve_calls, resolve_inherits

    def _package_of(n: Node):
        return pkg_of.get(n.id)

    resolve_calls(((cid, name, is_this) for f in files for cid, name, is_this in f.calls),
                  name_index, by_id, _package_of, seen, edges, "tree-sitter-java")
    resolve_inherits(((tid, base) for f in files for tid, base in f.bases),
                     name_index, by_id, _package_of, ("class", "interface"), seen, edges)

    ext: dict[str, Node] = {}
    for f in files:
        for mod_id, target in f.imports:
            internal = qual_index.get(target)
            if internal is not None:
                dst = internal.id
            else:
                e = ext.get(target)
                if e is None:
                    e = Node(id=f"ext::{target}", kind="external", name=target.rsplit(".", 1)[-1],
                             qualified_name=target, module=target, file_path="<external>",
                             start_line=0, end_line=0, embed_text=f"import {target}")
                    ext[target] = e
                dst = e.id
            if ("IMPORTS", mod_id, dst) not in seen:
                seen.add(("IMPORTS", mod_id, dst))
                edges.append(Edge("IMPORTS", mod_id, dst, EXTRACTED))
    nodes.extend(ext.values())
    return Graph(nodes=nodes, edges=edges)
