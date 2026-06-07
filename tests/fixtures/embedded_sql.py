"""A raw-SQL (sqlite) data layer — SQL embedded in Python strings, no ORM."""

import sqlite3


def init_db(conn):
    conn.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id));
        """
    )


def save_order(conn, user_id):
    conn.execute("INSERT INTO orders (user_id) VALUES (?)", (user_id,))


def recent_users(conn):
    return conn.execute("SELECT id, email FROM users ORDER BY id DESC").fetchall()
