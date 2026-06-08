"""Raw-SQL with a JOIN and qualified column refs, for the deeper-schema graph."""

import sqlite3


def init(conn):
    conn.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total NUMERIC);
        """
    )


def user_order_totals(conn):
    return conn.execute(
        "SELECT users.email, orders.total FROM users JOIN orders ON users.id = orders.user_id"
    ).fetchall()
