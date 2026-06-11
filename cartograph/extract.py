"""tree-sitter extraction: Python source -> graph nodes + edges.

Deterministic, local, no network. Structural edges (CONTAINS, INHERITS, IMPORTS)
are tagged EXTRACTED; call edges are name-matched heuristics tagged INFERRED — the
known precision gap that M3 (SCIP / stack-graphs) will close.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import tree_sitter_python
from tree_sitter import Language, Node as TSNode, Parser

from .model import EXTRACTED, INFERRED, Edge, Graph, Node

_LANGUAGE = Language(tree_sitter_python.language())
_PARSER = Parser(_LANGUAGE)

# Inline rationale markers become their own `rationale` nodes (WHY-mode retrieval).
_MARKER_RE = re.compile(r"#\s*(NOTE|WHY|HACK|XXX|TODO|FIXME|IMPORTANT)\b", re.IGNORECASE)

# SQL embedded in Python string literals (raw-SQL apps: sqlite/psycopg/etc.).
_SQL_RE = re.compile(
    r"\b(create\s+table|insert\s+into|delete\s+from|update\s+\w+\s+set|select\b[\s\S]*?\bfrom)\b",
    re.IGNORECASE,
)


def _text(src: bytes, node: TSNode) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", "replace")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _clean_docstring(raw: str) -> str:
    """Strip string prefixes/quotes from a docstring literal."""
    s = raw.strip()
    s = re.sub(r"^[rbuRBUfF]{0,3}", "", s)
    for q in ('"""', "'''", '"', "'"):
        if s.startswith(q) and s.endswith(q) and len(s) >= 2 * len(q):
            return s[len(q) : len(s) - len(q)].strip()
    return s.strip()


def _docstring(body: TSNode | None, src: bytes) -> str:
    if body is None:
        return ""
    for child in body.named_children:
        if child.type == "expression_statement" and child.named_children:
            first = child.named_children[0]
            if first.type == "string":
                return _clean_docstring(_text(src, first))
        return ""
    return ""


def _signature(src: bytes, def_node: TSNode, body: TSNode | None) -> str:
    """The header line(s): everything from `def`/`class` up to the body."""
    end = body.start_byte if body is not None else def_node.end_byte
    head = src[def_node.start_byte : end].decode("utf-8", "replace")
    return re.sub(r"\s+", " ", head).strip().rstrip(":").strip()


def _callee_name(func: TSNode, src: bytes) -> str | None:
    """The simple name being called: `f(...)` -> 'f', `x.y.z(...)` -> 'z'."""
    if func.type == "identifier":
        return _text(src, func)
    if func.type == "attribute":
        attr = func.child_by_field_name("attribute")
        if attr is not None:
            return _text(src, attr)
    return None


def module_qualified_name(rel_path: str) -> str:
    """`httpx/_transports/default.py` -> `httpx._transports.default`."""
    p = Path(rel_path)
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class _FileExtractor:
    """Walks one parsed file, emitting nodes and (resolved-later) references."""

    def __init__(self, src: bytes, rel_path: str, module: str):
        self.src = src
        self.rel_path = rel_path
        self.module = module
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        # Pending references resolved across the whole corpus in pass 2.
        self.calls: list[tuple[str, str, bool, int, int]] = []  # (caller_id, name, is_self, row, col)
        self.bases: list[tuple[str, str]] = []  # (class_id, base_name)
        self.imports: list[tuple[str, str]] = []  # (module_id, imported_name)
        self.sql_strings: list[tuple[str, str]] = []  # (owner_id, sql_text) — SQL in string literals

    # -- node factory ---------------------------------------------------------
    def _add(self, kind: str, name: str, qualified_name: str, node: TSNode, body: TSNode | None) -> Node:
        code = _text(self.src, node)
        sig = _signature(self.src, node, body) if kind in ("class", "function", "method") else ""
        doc = _docstring(body, self.src)
        embed_text = "\n".join(p for p in (f"{kind} {qualified_name}", sig, doc) if p).strip()
        start_line = node.start_point[0] + 1
        n = Node(
            # Line-suffixed so property getter/setter pairs (same qualified name) stay unique.
            id=f"{self.rel_path}::{qualified_name}#{start_line}",
            kind=kind,
            name=name,
            qualified_name=qualified_name,
            module=self.module,
            file_path=self.rel_path,
            start_line=start_line,
            end_line=node.end_point[0] + 1,
            signature=sig,
            docstring=doc,
            code=code,
            embed_text=embed_text,
            content_sha=_sha(code),
        )
        self.nodes.append(n)
        return n

    def _add_rationale(self, text: str, owner_id: str, node: TSNode) -> None:
        clean = text.lstrip("#").strip()
        rid = f"{self.rel_path}::rationale@{node.start_point[0] + 1}"
        self.nodes.append(
            Node(
                id=rid,
                kind="rationale",
                name=clean[:48],
                qualified_name=rid,
                module=self.module,
                file_path=self.rel_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=clean,
                code=text,
                embed_text=f"rationale {clean}",
                content_sha=_sha(text),
            )
        )
        self.edges.append(Edge("DOCUMENTS", rid, owner_id, EXTRACTED))

    def _scan_comments(self, node: TSNode, owner_id: str) -> None:
        """Marker comments attach directly to def/class nodes — capture them here."""
        for child in node.children:
            if child.type == "comment" and _MARKER_RE.match(_text(self.src, child)):
                self._add_rationale(_text(self.src, child), owner_id, child)

    # -- traversal ------------------------------------------------------------
    def run(self, root: TSNode) -> None:
        module_node = Node(
            id=f"{self.rel_path}::{self.module}",
            kind="module",
            name=self.module.rsplit(".", 1)[-1],
            qualified_name=self.module,
            module=self.module,
            file_path=self.rel_path,
            start_line=1,
            end_line=root.end_point[0] + 1,
            docstring=_docstring(root, self.src),
            embed_text=f"module {self.module}\n{_docstring(root, self.src)}".strip(),
            content_sha=_sha(_text(self.src, root)),
        )
        self.nodes.append(module_node)
        self._walk(root, parent_id=module_node.id, enclosing_qual=self.module, in_class=False)
        self._collect_sql(root, module_node.id)  # top-level SQL (skips def bodies)
        if self.sql_strings:
            module_node.extra["sql"] = [(oid, self.rel_path, txt) for oid, txt in self.sql_strings]

    def _walk(self, node: TSNode, parent_id: str, enclosing_qual: str, in_class: bool) -> None:
        """`enclosing_qual` is the qualified name of the lexical parent scope
        (module, class, or function); `in_class` is true only when the *direct*
        parent scope is a class body — that's what makes a def a method."""
        for child in node.children:
            t = child.type
            if t == "comment":
                if _MARKER_RE.match(_text(self.src, child)):
                    self._add_rationale(_text(self.src, child), parent_id, child)
            elif t == "import_statement":
                self._handle_import(child)
            elif t == "import_from_statement":
                self._handle_import_from(child)
            elif t == "decorated_definition":
                self._walk(child, parent_id, enclosing_qual, in_class)
            elif t == "class_definition":
                self._handle_class(child, parent_id, enclosing_qual)
            elif t == "function_definition":
                self._handle_function(child, parent_id, enclosing_qual, in_class)
            else:
                self._walk(child, parent_id, enclosing_qual, in_class)

    def _handle_class(self, node: TSNode, parent_id: str, enclosing_qual: str) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = _text(self.src, name_node)
        qual = f"{enclosing_qual}.{name}"
        body = node.child_by_field_name("body")
        cls = self._add("class", name, qual, node, body)
        self.edges.append(Edge("CONTAINS", parent_id, cls.id, EXTRACTED))
        self._scan_comments(node, cls.id)
        tablename = self._orm_tablename(body)
        if tablename:
            cls.extra["tablename"] = tablename  # bridged to a SQL table in build_graph
        supers = node.child_by_field_name("superclasses")
        if supers is not None:
            for base in supers.named_children:
                bname = _callee_name(base, self.src) if base.type == "attribute" else (
                    _text(self.src, base) if base.type == "identifier" else None
                )
                if bname:
                    self.bases.append((cls.id, bname))
        if body is not None:
            self._walk(body, parent_id=cls.id, enclosing_qual=qual, in_class=True)

    def _orm_tablename(self, body: TSNode | None) -> str | None:
        """`__tablename__ = "users"` / `db_table = "users"` in a class body -> 'users'."""
        if body is None:
            return None
        for child in body.children:
            stmt = child.named_children[0] if (child.type == "expression_statement" and child.named_children) else None
            if stmt is None or stmt.type != "assignment":
                continue
            left = stmt.child_by_field_name("left")
            right = stmt.child_by_field_name("right")
            if (left is not None and right is not None and left.type == "identifier"
                    and _text(self.src, left) in ("__tablename__", "db_table") and right.type == "string"):
                return _clean_docstring(_text(self.src, right))
        return None

    def _handle_function(self, node: TSNode, parent_id: str, enclosing_qual: str, in_class: bool) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = _text(self.src, name_node)
        kind = "method" if in_class else "function"
        qual = f"{enclosing_qual}.{name}"
        body = node.child_by_field_name("body")
        fn = self._add(kind, name, qual, node, body)
        self.edges.append(Edge("CONTAINS", parent_id, fn.id, EXTRACTED))
        self._scan_comments(node, fn.id)
        if body is not None:
            self._collect_calls(body, fn.id)
            self._collect_sql(body, fn.id)
            # Nested defs / classes own their own scope: a local `def helper()` inside
            # a method is a function `…Class.method.helper`, never a phantom method.
            self._walk(body, parent_id=fn.id, enclosing_qual=qual, in_class=False)

    def _collect_sql(self, node: TSNode, owner_id: str) -> None:
        """Record string literals that contain SQL, owned by the enclosing def/module."""
        if node.type == "string":
            text = _clean_docstring(_text(self.src, node))
            if _SQL_RE.search(text):
                self.sql_strings.append((owner_id, text))
            return
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                continue  # nested defs own their own SQL
            self._collect_sql(child, owner_id)

    def _collect_calls(self, node: TSNode, caller_id: str) -> None:
        if node.type == "call":
            func = node.child_by_field_name("function")
            if func is not None:
                # The identifier actually being called (rightmost name), with its position
                # so a real symbol resolver (Jedi) can do receiver-type goto.
                name_node = (func.child_by_field_name("attribute") if func.type == "attribute"
                             else func if func.type == "identifier" else None)
                if name_node is not None:
                    name = _text(self.src, name_node)
                    # A `self.x()` / `cls.x()` receiver lets us resolve to the caller's
                    # own class — disambiguating same-name methods (e.g. sync vs async).
                    is_self = False
                    if func.type == "attribute":
                        obj = func.child_by_field_name("object")
                        if obj is not None and obj.type == "identifier" and _text(self.src, obj) in ("self", "cls"):
                            is_self = True
                    row, col = name_node.start_point
                    self.calls.append((caller_id, name, is_self, row, col))
        for child in node.children:
            # Don't descend into nested function/class bodies; their calls belong to them.
            if child.type in ("function_definition", "class_definition"):
                continue
            self._collect_calls(child, caller_id)

    def _handle_import(self, node: TSNode) -> None:
        mod_id = f"{self.rel_path}::{self.module}"
        for child in node.named_children:
            target = child
            if child.type == "aliased_import":
                target = child.child_by_field_name("name") or child
            if target.type in ("dotted_name", "identifier"):
                top = _text(self.src, target).split(".")[0]
                self.imports.append((mod_id, top))

    def _handle_import_from(self, node: TSNode) -> None:
        # IMPORTS edges connect a module to imported *modules*, not to symbols —
        # recording symbol names mints noisy external stubs and mis-resolves them.
        mod_id = f"{self.rel_path}::{self.module}"
        mod_name_node = node.child_by_field_name("module_name")
        top = ""
        if mod_name_node is not None:
            top = _text(self.src, mod_name_node).lstrip(".").split(".")[0]
        if top:
            # `from pkg.mod import ...` / `from .mod import ...` -> import the module.
            self.imports.append((mod_id, top))
        else:
            # `from . import sub` -> the imported names are submodules; record those.
            for child in node.named_children:
                if child is mod_name_node:
                    continue
                target = child
                if child.type == "aliased_import":
                    target = child.child_by_field_name("name") or child
                if target.type in ("dotted_name", "identifier"):
                    self.imports.append((mod_id, _text(self.src, target).split(".")[0]))


def extract_source(source: str, rel_path: str, module: str | None = None) -> _FileExtractor:
    src = source.encode("utf-8")
    tree = _PARSER.parse(src)
    mod = module or module_qualified_name(rel_path)
    fx = _FileExtractor(src, rel_path, mod)
    fx.run(tree.root_node)
    return fx


def _resolve_calls_jedi(extractors, paths, pkg_parent, by_id, qual_index, heuristic_targets, add_call):
    """Resolve call edges with Jedi (receiver-type inference). Falls back to the
    heuristic only when Jedi returns nothing; if Jedi resolves a call to something
    outside the graph (stdlib/3rd-party), no edge is added (precision over noise)."""
    import jedi

    project = jedi.Project(str(pkg_parent))
    for fx, path in zip(extractors, paths):
        try:
            script = jedi.Script(code=fx.src.decode("utf-8", "replace"), path=str(path), project=project)
        except Exception:
            script = None
        src_lines = fx.src.split(b"\n")
        for caller_id, name, is_self, row, col in fx.calls:
            caller = by_id.get(caller_id)
            target_ids: list[str] | None = None
            jedi_decided = False
            if script is not None:
                # tree-sitter gives a byte column; Jedi wants a character column.
                line_bytes = src_lines[row] if row < len(src_lines) else b""
                char_col = len(line_bytes[:col].decode("utf-8", "replace"))
                try:
                    defs = script.goto(row + 1, char_col, follow_imports=True)
                except Exception:
                    defs = []
                if defs:
                    jedi_decided = True
                    target_ids = [qual_index[d.full_name].id for d in defs
                                  if d.full_name and d.full_name in qual_index]
            if target_ids is None:  # Jedi gave no opinion -> heuristic fallback
                target_ids = [c.id for c in heuristic_targets(caller, name, is_self)]
            for tid in target_ids:
                add_call(caller_id, tid, "jedi" if jedi_decided else "tree-sitter")


def extract_paths(paths: list[Path], root: Path, resolver: str = "heuristic") -> Graph:
    """Extract one or many files and resolve cross-file references into edges.

    `resolver`: 'heuristic' (name + self-class, no deps) or 'jedi' (receiver-type
    inference; needs the `resolve` extra)."""
    pkg_parent = root.parent if root.is_dir() else root.parent
    extractors: list[_FileExtractor] = []
    for path in paths:
        rel = path.relative_to(pkg_parent).as_posix() if path.is_relative_to(pkg_parent) else path.name
        module = module_qualified_name(rel)
        extractors.append(extract_source(path.read_text(encoding="utf-8", errors="replace"), rel, module))

    nodes: list[Node] = []
    name_index: dict[str, list[Node]] = {}
    module_index: dict[str, Node] = {}
    for fx in extractors:
        for n in fx.nodes:
            nodes.append(n)
            if n.kind in ("function", "method", "class"):
                name_index.setdefault(n.name, []).append(n)
            if n.kind == "module":
                module_index[n.qualified_name] = n

    edges: list[Edge] = list(_dedupe(e for fx in extractors for e in fx.edges))
    by_id = {n.id: n for n in nodes}
    qual_index: dict[str, Node] = {}
    for n in nodes:
        if n.kind in ("function", "method", "class") and n.qualified_name not in qual_index:
            qual_index[n.qualified_name] = n

    # Resolve calls (INFERRED). Prefer the caller's own class for `self.` calls, then
    # same-module candidates, before falling back to all name matches.
    seen: set[tuple[str, str, str]] = {(e.type, e.src, e.dst) for e in edges}

    def _class_of(node: Node) -> str | None:
        return node.qualified_name.rsplit(".", 1)[0] if node.kind == "method" else None

    def _heuristic_targets(caller: Node | None, name: str, is_self: bool) -> list[Node]:
        cands = name_index.get(name)
        if not cands:
            return []
        chosen: list[Node] | None = None
        if is_self and caller is not None and caller.kind == "method":
            caller_class = _class_of(caller)
            same_class = [c for c in cands if _class_of(c) == caller_class]
            if same_class:  # self.x() bound to this class's own method
                chosen = same_class
        if chosen is None:
            same = [c for c in cands if caller and c.module == caller.module]
            chosen = same or cands
        if len(chosen) > 8:  # avoid god-node fan-out on very common names
            return []
        return chosen

    def _add_call(caller_id: str, target_id: str, resolver_tag: str) -> None:
        if target_id == caller_id:
            return
        key = ("CALLS", caller_id, target_id)
        if key not in seen:
            seen.add(key)
            edges.append(Edge("CALLS", caller_id, target_id, INFERRED, resolver_tag))

    if resolver == "jedi":
        try:
            import jedi  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "resolver='jedi' needs the 'resolve' extra: uv sync --extra resolve"
            ) from exc
        _resolve_calls_jedi(extractors, paths, pkg_parent, by_id, qual_index,
                            _heuristic_targets, _add_call)
    else:
        for fx in extractors:
            for caller_id, name, is_self, _row, _col in fx.calls:
                caller = by_id.get(caller_id)
                for c in _heuristic_targets(caller, name, is_self):
                    _add_call(caller_id, c.id, "tree-sitter")

    # Resolve inheritance. A unique name match is deterministic (EXTRACTED); when
    # several same-named classes exist, at most one edge is right — every candidate
    # is a guess and must say so (INFERRED), per the confidence invariant.
    for fx in extractors:
        for cls_id, base in fx.bases:
            matches = [c for c in name_index.get(base, [])
                       if c.kind == "class" and c.id != cls_id]
            confidence = EXTRACTED if len(matches) == 1 else INFERRED
            for c in matches:
                key = ("INHERITS", cls_id, c.id)
                if key not in seen:
                    seen.add(key)
                    edges.append(Edge("INHERITS", cls_id, c.id, confidence))

    # Resolve imports: internal module if known, else an external module node.
    ext_nodes: dict[str, Node] = {}
    for fx in extractors:
        for mod_id, target in fx.imports:
            internal = _match_module(target, module_index)
            if internal is not None:
                dst = internal.id
            else:
                ext = ext_nodes.get(target)
                if ext is None:
                    ext = Node(
                        id=f"ext::{target}",
                        kind="external",
                        name=target,
                        qualified_name=target,
                        module=target,
                        file_path="<external>",
                        start_line=0,
                        end_line=0,
                        embed_text=f"module {target}",
                    )
                    ext_nodes[target] = ext
                dst = ext.id
            key = ("IMPORTS", mod_id, dst)
            if key not in seen:
                seen.add(key)
                edges.append(Edge("IMPORTS", mod_id, dst, EXTRACTED))
    nodes.extend(ext_nodes.values())

    return Graph(nodes=nodes, edges=edges)


def _match_module(target: str, module_index: dict[str, Node]) -> Node | None:
    if target in module_index:
        return module_index[target]
    for qual, node in module_index.items():
        if qual.endswith("." + target) or qual.rsplit(".", 1)[-1] == target:
            return node
    return None


def _dedupe(edges):
    seen: set[tuple[str, str, str]] = set()
    for e in edges:
        key = (e.type, e.src, e.dst)
        if key not in seen:
            seen.add(key)
            yield e
