package database

import (
    "database/sql"
    _ "github.com/lib/pq"
)

type DB struct {{
    conn *sql.DB
}}

func NewDB(connStr string) (*DB, error) {{
    conn, err := sql.Open("postgres", connStr)
    if err != nil {{
        return nil, err
    }}
    return &DB{{conn: conn}}, nil
}}

func (db *DB) Query(query string, args ...interface{{}}) (*sql.Rows, error) {{
    return db.conn.Query(query, args...)
}}

func (db *DB) Close() error {{
    return db.conn.Close()
}}