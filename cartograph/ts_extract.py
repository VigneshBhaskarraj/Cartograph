"""tree-sitter extraction for TypeScript/JavaScript -> the same graph model.

Proves the architecture extends past Python (SPEC non-goal: no breadth yet, but the
seam should be clean). Optional `ts` extra (tree-sitter-typescript +
tree-sitter-javascript). Handles .ts/.tsx/.js/.jsx/.mjs/.cjs — the JS grammar shares
the node-type vocabulary this walker uses (TS-only kinds like interfaces simply never
appear in JS trees). Produces module/class/interface/function/method nodes with
CONTAINS, INHERITS (extends/implements), IMPORTS, and heuristic INFERRED CALLS —
mirroring the Python extractor. No network.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .model import EXTRACTED, INFERRED, Edge, Graph, Node

JS_SUFFIXES = (".js", ".jsx", ".mjs", ".cjs")


def _parsers(need_js: bool):
    """suffix -> Parser. The JS grammar is imported only when a JS file exists, so
    a TypeScript-only environment never needs it."""
    import tree_sitter_typescript as tsts
    from tree_sitter import Language, Parser

    ts = Parser(Language(tsts.language_typescript()))
    tsx = Parser(Language(tsts.language_tsx()))
    out = {".ts": ts, ".tsx": tsx}
    if need_js:
        import tree_sitter_javascript as tsjs

        js = Parser(Language(tsjs.language()))
        out.update({s: js for s in JS_SUFFIXES})
    return out


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8", "replace")).hexdigest()


def module_name(rel_path: str) -> str:
    p = Path(rel_path)
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] in ("index",):
        parts = parts[:-1]
    return ".".join(parts)


class _TsFile:
    def __init__(self, src: bytes, rel_path: str):
        self.src = src
        self.rel_path = rel_path
        self.module = module_name(rel_path)
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.calls: list[tuple[str, str, bool]] = []   # (caller_id, name, is_this)
        self.bases: list[tuple[str, str]] = []          # (class_id, base_name)
        self.imports: list[tuple[str, str]] = []        # (module_id, source)

    def _text(self, n) -> str:
        return self.src[n.start_byte:n.end_byte].decode("utf-8", "replace")

    def _first_line(self, n) -> str:
        return re.sub(r"\s+", " ", self._text(n).split("\n", 1)[0]).strip().rstrip("{").strip()

    def _node(self, kind: str, name: str, qual: str, n) -> Node:
        line = n.start_point[0] + 1
        sig = self._first_line(n)
        node = Node(
            id=f"{self.rel_path}::{qual}#{line}", kind=kind, name=name, qualified_name=qual,
            module=self.module, file_path=self.rel_path, start_line=line, end_line=n.end_point[0] + 1,
            signature=sig, embed_text=f"{kind} {qual}\n{sig}".strip(), content_sha=_sha(self._text(n)),
        )
        self.nodes.append(node)
        return node

    def run(self, root) -> None:
        self.module_node = Node(
            id=f"{self.rel_path}::{self.module}", kind="module", name=self.module.rsplit(".", 1)[-1],
            qualified_name=self.module, module=self.module, file_path=self.rel_path,
            start_line=1, end_line=root.end_point[0] + 1, embed_text=f"module {self.module}",
            content_sha=_sha(self._text(root)),
        )
        self.nodes.append(self.module_node)
        self._walk(root, self.module_node.id, None)

    def _decl(self, n):  # unwrap export/default wrappers
        if n.type in ("export_statement",):
            d = n.child_by_field_name("declaration")
            return d if d is not None else n
        return n

    def _walk(self, node, parent_id: str, class_qual: str | None) -> None:
        for child in node.children:
            c = self._decl(child)
            t = c.type
            if t in ("function_declaration", "generator_function_declaration"):
                self._fn(c, parent_id, class_qual)
            elif t in ("class_declaration", "abstract_class_declaration"):
                self._class(c, parent_id)
            elif t == "interface_declaration":
                nm = c.child_by_field_name("name")
                if nm is not None:
                    iface = self._node("interface", self._text(nm), f"{self.module}.{self._text(nm)}", c)
                    self.edges.append(Edge("CONTAINS", parent_id, iface.id, EXTRACTED))
            elif t == "lexical_declaration" or t == "variable_declaration":
                self._arrow_consts(c, parent_id, class_qual)
            elif t == "expression_statement":
                if self._assignment_fn(c, parent_id) is None:
                    # Walk only when no node was minted: descending into a matched
                    # assignment's function body used to leak its nested
                    # declarations to module scope (G5-C4).
                    self._walk(c, parent_id, class_qual)
            elif t == "import_statement":
                src = c.child_by_field_name("source")
                if src is not None:
                    self.imports.append((self.module_node.id, self._text(src).strip("\"'`")))
            else:
                self._walk(c, parent_id, class_qual)

    # CommonJS noise segments stripped from assignment paths: `exports.init`,
    # `module.exports.x`, and `Route.prototype.dispatch` name the function/method,
    # not a real object hierarchy.
    _PATH_NOISE = ("module", "exports", "prototype")

    def _assignment_fn(self, stmt, parent_id) -> Node | None:
        """CommonJS-style definitions: `exports.f = function`, `app.x = () => …`,
        `Foo.prototype.m = function` — most pre-ES6 Node code defines this way.
        Returns the minted node (callers skip re-walking the expression) or None."""
        expr = next((c for c in stmt.named_children if c.type == "assignment_expression"), None)
        if expr is None:
            return None
        left, right = expr.child_by_field_name("left"), expr.child_by_field_name("right")
        if left is None or right is None or right.type not in ("function_expression", "function", "arrow_function"):
            return None
        if left.type not in ("member_expression", "identifier"):
            return None  # obj[key] = fn etc. — no stable dotted path to name it by
        raw = self._text(left).split(".")
        if raw and raw[0] == "this":
            return None  # this.x = fn inside a constructor isn't a module-level symbol
        segments = [s for s in raw if s and s.isidentifier()]
        if len(segments) != len(raw):
            return None  # any non-identifier segment means a computed/odd path
        is_proto = "prototype" in segments
        cleaned = [s for s in segments if s not in self._PATH_NOISE]
        if not cleaned:  # bare `module.exports = function` — use the fn's own name if any
            nm = right.child_by_field_name("name")
            if nm is None:
                return None
            cleaned = [self._text(nm)]
        name = cleaned[-1]
        kind = "method" if is_proto and len(cleaned) > 1 else "function"
        qual = f"{self.module}." + ".".join(cleaned)
        fn = self._node(kind, name, qual, expr)
        self.edges.append(Edge("CONTAINS", parent_id, fn.id, EXTRACTED))
        body = right.child_by_field_name("body")
        if body is not None:
            self._collect_calls(body, fn.id)
        return fn

    def _arrow_consts(self, node, parent_id, class_qual) -> None:
        for vd in node.children:
            if vd.type != "variable_declarator":
                continue
            name_n = vd.child_by_field_name("name")
            val = vd.child_by_field_name("value")
            if name_n is None or val is None:
                continue
            if val.type in ("arrow_function", "function_expression", "function"):
                qual = f"{self.module}.{self._text(name_n)}"
                fn = self._node("function", self._text(name_n), qual, vd)
                self.edges.append(Edge("CONTAINS", parent_id, fn.id, EXTRACTED))
                body = val.child_by_field_name("body")
                if body is not None:
                    self._collect_calls(body, fn.id)
            elif val.type == "call_expression":  # const x = require('pkg')
                callee = val.child_by_field_name("function")
                args = val.child_by_field_name("arguments")
                if (callee is not None and self._text(callee) == "require"
                        and args is not None and args.named_children
                        and args.named_children[0].type == "string"):
                    self.imports.append(
                        (self.module_node.id, self._text(args.named_children[0]).strip("\"'`")))

    def _fn(self, node, parent_id, class_qual) -> None:
        nm = node.child_by_field_name("name")
        if nm is None:
            return
        kind = "method" if class_qual else "function"
        qual = f"{class_qual}.{self._text(nm)}" if class_qual else f"{self.module}.{self._text(nm)}"
        fn = self._node(kind, self._text(nm), qual, node)
        self.edges.append(Edge("CONTAINS", parent_id, fn.id, EXTRACTED))
        body = node.child_by_field_name("body")
        if body is not None:
            self._collect_calls(body, fn.id)

    def _class(self, node, parent_id) -> None:
        nm = node.child_by_field_name("name")
        if nm is None:
            return
        name = self._text(nm)
        qual = f"{self.module}.{name}"
        cls = self._node("class", name, qual, node)
        self.edges.append(Edge("CONTAINS", parent_id, cls.id, EXTRACTED))
        for h in node.children:
            if h.type == "class_heritage":
                # The recursive walk handles both grammars (TS wraps bases in
                # extends_clause/implements_clause nodes; JS puts the expression
                # directly under class_heritage) — but it must NOT descend into
                # type_arguments: `extends Component<Props, State>` names ONE
                # base, not three (G5-C1).
                for desc in _iter(h, skip=("type_arguments",)):
                    if desc.type in ("identifier", "type_identifier"):
                        self.bases.append((cls.id, self._text(desc)))
        body = node.child_by_field_name("body")
        if body is not None:
            for m in body.children:
                if m.type == "method_definition":
                    mn = m.child_by_field_name("name")
                    if mn is not None:
                        mq = f"{qual}.{self._text(mn)}"
                        meth = self._node("method", self._text(mn), mq, m)
                        self.edges.append(Edge("CONTAINS", cls.id, meth.id, EXTRACTED))
                        mb = m.child_by_field_name("body")
                        if mb is not None:
                            self._collect_calls(mb, meth.id)
                elif m.type in ("public_field_definition", "field_definition"):
                    # `handleClick = () => { … }` — the standard React/handler
                    # idiom is a method in all but grammar node type (G5-C4).
                    mn = m.child_by_field_name("name")
                    val = m.child_by_field_name("value")
                    if (mn is not None and val is not None
                            and val.type in ("arrow_function", "function_expression", "function")):
                        mq = f"{qual}.{self._text(mn)}"
                        meth = self._node("method", self._text(mn), mq, m)
                        self.edges.append(Edge("CONTAINS", cls.id, meth.id, EXTRACTED))
                        mb = val.child_by_field_name("body")
                        if mb is not None:
                            self._collect_calls(mb, meth.id)

    def _collect_calls(self, node, caller_id: str) -> None:
        if node.type == "call_expression":
            fn = node.child_by_field_name("function")
            if fn is not None:
                name, is_this = None, False
                if fn.type == "identifier":
                    name = self._text(fn)
                elif fn.type == "member_expression":
                    prop = fn.child_by_field_name("property")
                    obj = fn.child_by_field_name("object")
                    if prop is not None:
                        name = self._text(prop)
                        is_this = obj is not None and obj.type == "this"
                if name:
                    self.calls.append((caller_id, name, is_this))
        for child in node.children:
            # Arrows are NOT skipped: `items.forEach(i => doWork())` must credit
            # doWork to the enclosing function, exactly as Python does for lambdas
            # and as function-expression callbacks already did here (G5-C4).
            if child.type in ("function_declaration", "class_declaration", "method_definition"):
                continue
            self._collect_calls(child, caller_id)


def _iter(node, skip: tuple[str, ...] = ()):
    for c in node.children:
        if c.type in skip:
            continue
        yield c
        yield from _iter(c, skip)


def extract_ts_paths(paths: list[Path], root: Path) -> Graph:
    parsers = _parsers(need_js=any(p.suffix in JS_SUFFIXES for p in paths))
    pkg_parent = root.parent
    files: list[_TsFile] = []
    for path in paths:
        rel = path.relative_to(pkg_parent).as_posix() if path.is_relative_to(pkg_parent) else path.name
        src = path.read_bytes()
        parser = parsers[path.suffix]
        fx = _TsFile(src, rel)
        tree = parser.parse(src)
        if tree.root_node.has_error:
            import warnings  # tree-sitter never raises; partial graphs must not be silent (G5-C6)
            warnings.warn(f"syntax errors in {rel}; its graph may be partial", stacklevel=2)
        fx.run(tree.root_node)
        files.append(fx)

    nodes: list[Node] = [n for f in files for n in f.nodes]
    name_index: dict[str, list[Node]] = {}
    module_index: dict[str, Node] = {}
    for n in nodes:
        if n.kind in ("function", "method", "class", "interface"):
            name_index.setdefault(n.name, []).append(n)
        if n.kind == "module":
            module_index[n.qualified_name] = n
    by_id = {n.id: n for n in nodes}
    edges: list[Edge] = list({(e.type, e.src, e.dst): e for f in files for e in f.edges}.values())
    seen = {(e.type, e.src, e.dst) for e in edges}

    def _class_of(n: Node) -> str | None:
        return n.qualified_name.rsplit(".", 1)[0] if n.kind == "method" else None

    for f in files:
        for caller_id, name, is_this in f.calls:
            cands = name_index.get(name)
            if not cands:
                continue
            caller = by_id.get(caller_id)
            chosen = None
            if is_this and caller is not None and caller.kind == "method":
                same = [c for c in cands if _class_of(c) == _class_of(caller)]
                if same:
                    chosen = same
            if chosen is None:
                samemod = [c for c in cands if caller and c.module == caller.module]
                chosen = samemod or cands
            if len(chosen) > 8:
                continue
            for c in chosen:
                if c.id != caller_id and ("CALLS", caller_id, c.id) not in seen:
                    seen.add(("CALLS", caller_id, c.id))
                    edges.append(Edge("CALLS", caller_id, c.id, INFERRED, "tree-sitter-ts"))

    # G5-C2: same-module resolution is EXTRACTED; cross-module (even unique) is a
    # guess — INFERRED — and same-module shadows cross-module. See extract.py.
    for f in files:
        for cls_id, base in f.bases:
            matches = [c for c in name_index.get(base, [])
                       if c.kind in ("class", "interface") and c.id != cls_id]
            cls = by_id.get(cls_id)
            same_mod = [c for c in matches if cls is not None and c.module == cls.module]
            confidence = EXTRACTED if len(same_mod) == 1 else INFERRED
            for c in (same_mod or matches):
                if ("INHERITS", cls_id, c.id) not in seen:
                    seen.add(("INHERITS", cls_id, c.id))
                    edges.append(Edge("INHERITS", cls_id, c.id, confidence))

    ext: dict[str, Node] = {}
    for f in files:
        for mod_id, source in f.imports:
            # Name-boundary match: './utils' must bind to `pkg.utils`, never to a
            # module that merely ends with the substring (`pkg.statsutils`).
            src = source.split("/")[-1]
            internal = next(
                (m for q, m in module_index.items() if q == src or q.endswith("." + src)), None)
            if internal is not None:
                dst = internal.id
            else:
                e = ext.get(source)
                if e is None:
                    e = Node(id=f"ext::{source}", kind="external", name=source.split("/")[-1],
                             qualified_name=source, module=source, file_path="<external>",
                             start_line=0, end_line=0, embed_text=f"module {source}")
                    ext[source] = e
                dst = e.id
            if ("IMPORTS", mod_id, dst) not in seen:
                seen.add(("IMPORTS", mod_id, dst))
                edges.append(Edge("IMPORTS", mod_id, dst, EXTRACTED))
    nodes.extend(ext.values())
    return Graph(nodes=nodes, edges=edges)
