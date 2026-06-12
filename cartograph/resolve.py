"""Shared reference-resolution pass for all language extractors (G5-C5).

Four extractors used to carry near-identical ~65-line copies of this logic —
exactly where the G5-C2/C3 bug class bred: a fix in one file didn't propagate,
and "same module" silently meant "same file" in Go/Java. The language
differences are injected instead of copied: `scope_of` abstracts the lexical
resolution unit (module for Python/TS, package for Go/Java) and `kinds` the
node kinds a base name may bind to.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .model import EXTRACTED, INFERRED, Edge, Node

# Above this many same-name candidates the name is a god-name (`get`, `run`):
# emitting 9+ guesses pollutes the graph leg worse than emitting none.
FANOUT_CAP = 8


def class_of(n: Node) -> str | None:
    """Owning class qual of a method node — receiver-call binding (`self.x()`,
    `this.x()`, Go receiver methods)."""
    return n.qualified_name.rsplit(".", 1)[0] if n.kind == "method" else None


def make_heuristic_targets(name_index: dict[str, list[Node]],
                           scope_of: Callable[[Node], object],
                           fanout_cap: int = FANOUT_CAP):
    """Tiered candidate selection for one call site: a receiver call binds to the
    caller's own class when it can; otherwise same lexical scope shadows
    everything; otherwise every name match — capped."""
    def targets(caller: Node | None, name: str, is_receiver: bool) -> list[Node]:
        cands = name_index.get(name)
        if not cands:
            return []
        chosen: list[Node] | None = None
        if is_receiver and caller is not None and caller.kind == "method":
            same_class = [c for c in cands if class_of(c) == class_of(caller)]
            if same_class:
                chosen = same_class
        if chosen is None:
            same_scope = ([c for c in cands if scope_of(c) == scope_of(caller)]
                          if caller is not None else [])
            chosen = same_scope or cands
        if len(chosen) > fanout_cap:
            return []
        return chosen
    return targets


def resolve_calls(call_sites: Iterable[tuple[str, str, bool]],
                  name_index: dict[str, list[Node]], by_id: dict[str, Node],
                  scope_of: Callable[[Node], object],
                  seen: set, edges: list[Edge], resolver_tag: str,
                  fanout_cap: int = FANOUT_CAP) -> None:
    """Append INFERRED CALLS edges for (caller_id, name, is_receiver) sites."""
    targets = make_heuristic_targets(name_index, scope_of, fanout_cap)
    for caller_id, name, is_receiver in call_sites:
        caller = by_id.get(caller_id)
        for c in targets(caller, name, is_receiver):
            if c.id == caller_id:
                continue
            key = ("CALLS", caller_id, c.id)
            if key not in seen:
                seen.add(key)
                edges.append(Edge("CALLS", caller_id, c.id, INFERRED, resolver_tag))


def resolve_inherits(bases: Iterable[tuple[str, str]],
                     name_index: dict[str, list[Node]], by_id: dict[str, Node],
                     scope_of: Callable[[Node], object], kinds: tuple[str, ...],
                     seen: set, edges: list[Edge]) -> None:
    """Append INHERITS edges for (type_id, base_name) pairs.

    Confidence policy (G5-C2/C3): a base resolving within the type's own lexical
    scope is deterministic — EXTRACTED — and shadows cross-scope matches. A
    cross-scope match, even a unique name, is a guess (the real base may be
    external and unindexed: `class User(models.Model)` must not pin a random
    local `Model` with top confidence) — INFERRED.
    """
    for type_id, base in bases:
        matches = [c for c in name_index.get(base, [])
                   if c.kind in kinds and c.id != type_id]
        node = by_id.get(type_id)
        same_scope = [c for c in matches
                      if node is not None and scope_of(c) == scope_of(node)]
        confidence = EXTRACTED if len(same_scope) == 1 else INFERRED
        for c in (same_scope or matches):
            key = ("INHERITS", type_id, c.id)
            if key not in seen:
                seen.add(key)
                edges.append(Edge("INHERITS", type_id, c.id, confidence))
