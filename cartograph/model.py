"""In-memory graph model shared by extraction, storage, and retrieval.

These dataclasses are the contract between the extractor (produces them) and the
store (persists them). Keeping them plain and serializable means the graph — not
source files — is the single source of truth at query time (see SPEC invariants).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Node kinds. A single node table distinguished by `kind` (see PLAN.md schema).
KINDS = ("module", "class", "function", "method", "rationale", "external")

# Edge types map 1:1 onto Kuzu REL tables.
EDGE_TYPES = ("CALLS", "INHERITS", "IMPORTS", "CONTAINS", "DOCUMENTS")

# Confidence tags carried by every edge (SPEC: EXTRACTED vs INFERRED).
EXTRACTED = "EXTRACTED"  # deterministic structure (containment, inheritance, imports)
INFERRED = "INFERRED"  # heuristic, name-matched (call edges before M3 symbol resolution)


@dataclass
class Node:
    """One code object or rationale comment."""

    id: str  # stable id: "<file_path>::<qualified_name>"
    kind: str
    name: str
    qualified_name: str
    module: str
    file_path: str
    start_line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    code: str = ""
    embed_text: str = ""
    content_sha: str = ""
    embedding: list[float] | None = None


@dataclass
class Edge:
    """A typed, confidence-tagged relationship between two node ids."""

    type: str  # one of EDGE_TYPES
    src: str
    dst: str
    confidence: str = EXTRACTED
    resolver: str = "tree-sitter"


@dataclass
class Graph:
    """A bundle of nodes and edges produced by the extractor."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
