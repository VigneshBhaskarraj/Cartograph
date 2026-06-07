"""Kuzu store: schema, load, and graph queries.

One embedded file holds the property graph. Embeddings are stored on the node and
read back for brute-force cosine retrieval (offline, deterministic, no extension
download). HNSW is a later speed optimization that does not change recall.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import kuzu

from .model import EDGE_TYPES, Edge, Graph, Node

DEFAULT_DIM = 768


def schema_ddl(dim: int = DEFAULT_DIM) -> list[str]:
    return [
        f"""
        CREATE NODE TABLE CodeNode (
            id STRING, kind STRING, name STRING, qualified_name STRING, module STRING,
            file_path STRING, start_line INT64, end_line INT64,
            signature STRING, docstring STRING, code STRING, embed_text STRING,
            embedding FLOAT[{dim}], content_sha STRING,
            PRIMARY KEY (id)
        )
        """,
        "CREATE REL TABLE CALLS (FROM CodeNode TO CodeNode, confidence STRING, resolver STRING)",
        "CREATE REL TABLE INHERITS (FROM CodeNode TO CodeNode, confidence STRING)",
        "CREATE REL TABLE IMPORTS (FROM CodeNode TO CodeNode, confidence STRING)",
        "CREATE REL TABLE CONTAINS (FROM CodeNode TO CodeNode)",
        "CREATE REL TABLE DOCUMENTS (FROM CodeNode TO CodeNode)",
        # Key/value metadata (e.g. which embedder produced the vectors) so readers
        # can reconstruct the matching query-time embedder. Keeps the graph the
        # single source of truth — no sidecar files.
        "CREATE NODE TABLE Meta (key STRING, value STRING, PRIMARY KEY (key))",
    ]


class Store:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = kuzu.Database(str(self.path))
        self.conn = kuzu.Connection(self.db)

    def close(self) -> None:
        self.conn.close()
        self.db.close()

    @classmethod
    def create(cls, db_path: str | Path, dim: int = DEFAULT_DIM, overwrite: bool = False) -> "Store":
        p = Path(db_path)
        if overwrite and p.exists():
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        store = cls(p)
        store.create_schema(dim)
        return store

    def create_schema(self, dim: int = DEFAULT_DIM) -> None:
        for stmt in schema_ddl(dim):
            self.conn.execute(stmt)

    def table_names(self) -> set[str]:
        res = self.conn.execute("CALL show_tables() RETURN name")
        names: set[str] = set()
        while res.has_next():
            names.add(res.get_next()[0])
        return names

    # -- loading --------------------------------------------------------------
    def load(self, graph: Graph, dim: int = DEFAULT_DIM) -> None:
        zero = [0.0] * dim
        for n in graph.nodes:
            emb = n.embedding if n.embedding is not None else zero
            self.conn.execute(
                """
                CREATE (c:CodeNode {
                    id: $id, kind: $kind, name: $name, qualified_name: $qn, module: $module,
                    file_path: $fp, start_line: $sl, end_line: $el,
                    signature: $sig, docstring: $doc, code: $code, embed_text: $et,
                    embedding: $emb, content_sha: $sha
                })
                """,
                {
                    "id": n.id, "kind": n.kind, "name": n.name, "qn": n.qualified_name,
                    "module": n.module, "fp": n.file_path, "sl": n.start_line, "el": n.end_line,
                    "sig": n.signature, "doc": n.docstring, "code": n.code, "et": n.embed_text,
                    "emb": emb, "sha": n.content_sha,
                },
            )
        for e in graph.edges:
            self._insert_edge(e)

    def _insert_edge(self, e: Edge) -> None:
        if e.type == "CALLS":
            q = ("MATCH (a:CodeNode {id:$s}),(b:CodeNode {id:$d}) "
                 "CREATE (a)-[:CALLS {confidence:$c, resolver:$r}]->(b)")
            self.conn.execute(q, {"s": e.src, "d": e.dst, "c": e.confidence, "r": e.resolver})
        elif e.type in ("INHERITS", "IMPORTS"):
            q = (f"MATCH (a:CodeNode {{id:$s}}),(b:CodeNode {{id:$d}}) "
                 f"CREATE (a)-[:{e.type} {{confidence:$c}}]->(b)")
            self.conn.execute(q, {"s": e.src, "d": e.dst, "c": e.confidence})
        else:  # CONTAINS, DOCUMENTS
            q = (f"MATCH (a:CodeNode {{id:$s}}),(b:CodeNode {{id:$d}}) "
                 f"CREATE (a)-[:{e.type}]->(b)")
            self.conn.execute(q, {"s": e.src, "d": e.dst})

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "MERGE (m:Meta {key:$k}) SET m.value = $v", {"k": key, "v": value}
        )

    def get_meta(self, key: str) -> str | None:
        try:
            res = self.conn.execute("MATCH (m:Meta {key:$k}) RETURN m.value", {"k": key})
        except RuntimeError:
            return None  # older DB without the Meta table
        return res.get_next()[0] if res.has_next() else None

    def set_embedding(self, node_id: str, vector: list[float]) -> None:
        self.conn.execute(
            "MATCH (c:CodeNode {id:$id}) SET c.embedding = $emb",
            {"id": node_id, "emb": vector},
        )

    # -- queries --------------------------------------------------------------
    def get_node(self, node_id: str) -> dict | None:
        res = self.conn.execute(
            "MATCH (c:CodeNode {id:$id}) "
            "RETURN c.id, c.kind, c.name, c.qualified_name, c.file_path, c.start_line, c.signature, c.docstring",
            {"id": node_id},
        )
        if res.has_next():
            r = res.get_next()
            return {"id": r[0], "kind": r[1], "name": r[2], "qualified_name": r[3],
                    "file_path": r[4], "start_line": r[5], "signature": r[6], "docstring": r[7]}
        return None

    def all_nodes_text(self) -> list[dict]:
        """id + text fields for lexical and (re-)embedding. Excludes the vector."""
        res = self.conn.execute(
            "MATCH (c:CodeNode) RETURN c.id, c.kind, c.name, c.qualified_name, c.embed_text, c.docstring"
        )
        out = []
        while res.has_next():
            r = res.get_next()
            out.append({"id": r[0], "kind": r[1], "name": r[2], "qualified_name": r[3],
                        "embed_text": r[4], "docstring": r[5]})
        return out

    def all_embeddings(self) -> tuple[list[str], list[list[float]]]:
        res = self.conn.execute("MATCH (c:CodeNode) RETURN c.id, c.embedding")
        ids: list[str] = []
        vecs: list[list[float]] = []
        while res.has_next():
            r = res.get_next()
            ids.append(r[0])
            vecs.append(list(r[1]) if r[1] is not None else [])
        return ids, vecs

    def all_edges(self) -> list[tuple[str, str]]:
        """Every (src, dst) across all rel tables — for in-memory graph algorithms."""
        out: list[tuple[str, str]] = []
        for et in EDGE_TYPES:
            res = self.conn.execute(f"MATCH (a:CodeNode)-[:{et}]->(b:CodeNode) RETURN a.id, b.id")
            while res.has_next():
                r = res.get_next()
                out.append((r[0], r[1]))
        return out

    def neighbors(self, node_id: str, hops: int = 1) -> list[str]:
        res = self.conn.execute(
            f"MATCH (a:CodeNode {{id:$id}})-[*1..{hops}]-(b:CodeNode) RETURN DISTINCT b.id",
            {"id": node_id},
        )
        out = []
        while res.has_next():
            out.append(res.get_next()[0])
        return out

    def shortest_path(self, src: str, dst: str, max_hops: int = 8) -> list[str]:
        res = self.conn.execute(
            f"MATCH p = (a:CodeNode {{id:$s}})-[* SHORTEST 1..{max_hops}]-(b:CodeNode {{id:$d}}) "
            f"RETURN [n IN nodes(p) | n.id] LIMIT 1",
            {"s": src, "d": dst},
        )
        if res.has_next():
            return list(res.get_next()[0])
        return []

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        res = self.conn.execute("MATCH (c:CodeNode) RETURN c.kind, count(*)")
        while res.has_next():
            r = res.get_next()
            out[f"node:{r[0]}"] = r[1]
        for et in EDGE_TYPES:
            res = self.conn.execute(f"MATCH ()-[r:{et}]->() RETURN count(r)")
            out[f"edge:{et}"] = res.get_next()[0] if res.has_next() else 0
        return out
