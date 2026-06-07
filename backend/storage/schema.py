
import sqlite3


DATABASE_SCHEMA_VERSION = 5


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    file_count INTEGER NOT NULL DEFAULT 0
        CHECK (file_count >= 0),

    chapter_count INTEGER NOT NULL DEFAULT 0
        CHECK (chapter_count >= 0),

    chunk_count INTEGER NOT NULL DEFAULT 0
        CHECK (chunk_count >= 0)
);


CREATE TABLE IF NOT EXISTS source_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    project_id TEXT NOT NULL,

    file_order INTEGER NOT NULL
        CHECK (file_order > 0),

    file_name TEXT NOT NULL,

    source_path TEXT NOT NULL,
    processed_path TEXT NOT NULL,

    size_bytes INTEGER NOT NULL
        CHECK (size_bytes >= 0),

    original_character_count INTEGER NOT NULL
        CHECK (original_character_count >= 0),

    original_line_count INTEGER NOT NULL
        CHECK (original_line_count >= 0),

    processed_character_count INTEGER NOT NULL
        CHECK (processed_character_count >= 0),

    processed_line_count INTEGER NOT NULL
        CHECK (processed_line_count >= 0),

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (project_id)
        REFERENCES projects(id)
        ON DELETE CASCADE,

    UNIQUE (project_id, file_order)
);


CREATE TABLE IF NOT EXISTS chapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    source_file_id INTEGER NOT NULL,

    chapter_order INTEGER NOT NULL
        CHECK (chapter_order > 0),

    chapter_number INTEGER,
    chapter_title TEXT,
    full_title TEXT,

    part_order INTEGER,
    part_title TEXT,

    volume_order INTEGER,
    volume_title TEXT,

    start_character INTEGER NOT NULL
        CHECK (start_character >= 0),

    end_character INTEGER NOT NULL
        CHECK (end_character >= start_character),

    character_count INTEGER NOT NULL
        CHECK (
            character_count
            = end_character - start_character
        ),

    is_detected INTEGER NOT NULL DEFAULT 0
        CHECK (is_detected IN (0, 1)),

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (source_file_id)
        REFERENCES source_files(id)
        ON DELETE CASCADE,

    UNIQUE (source_file_id, chapter_order)
);


CREATE TABLE IF NOT EXISTS text_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    source_file_id INTEGER NOT NULL,
    chapter_id INTEGER NOT NULL,

    chunk_id TEXT NOT NULL,

    global_order INTEGER NOT NULL
        CHECK (global_order > 0),

    chunk_order_in_chapter INTEGER NOT NULL
        CHECK (chunk_order_in_chapter > 0),

    start_character INTEGER NOT NULL
        CHECK (start_character >= 0),

    end_character INTEGER NOT NULL
        CHECK (end_character >= start_character),

    character_count INTEGER NOT NULL
        CHECK (
            character_count
            = end_character - start_character
        ),

    paragraph_start INTEGER NOT NULL
        CHECK (paragraph_start > 0),

    paragraph_end INTEGER NOT NULL
        CHECK (paragraph_end >= paragraph_start),

    text TEXT NOT NULL,

    is_chapter_start INTEGER NOT NULL DEFAULT 0
        CHECK (is_chapter_start IN (0, 1)),

    is_chapter_end INTEGER NOT NULL DEFAULT 0
        CHECK (is_chapter_end IN (0, 1)),

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (source_file_id)
        REFERENCES source_files(id)
        ON DELETE CASCADE,

    FOREIGN KEY (chapter_id)
        REFERENCES chapters(id)
        ON DELETE CASCADE,

    UNIQUE (source_file_id, chunk_id),
    UNIQUE (chapter_id, chunk_order_in_chapter)
);


CREATE TABLE IF NOT EXISTS narrative_analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    project_id TEXT NOT NULL,

    provider TEXT NOT NULL,
    model TEXT NOT NULL,

    prompt_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,

    status TEXT NOT NULL DEFAULT 'queued',
    error_message TEXT,

    total_chunks INTEGER NOT NULL DEFAULT 0,
    processed_chunks INTEGER NOT NULL DEFAULT 0,
    successful_chunks INTEGER NOT NULL DEFAULT 0,
    partial_chunks INTEGER NOT NULL DEFAULT 0,
    failed_chunks INTEGER NOT NULL DEFAULT 0,
    cached_chunks INTEGER NOT NULL DEFAULT 0,
    cached_layers INTEGER NOT NULL DEFAULT 0,

    current_chunk_id TEXT,
    request_json TEXT,

    started_at TEXT,
    finished_at TEXT,
    heartbeat_at TEXT,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (project_id)
        REFERENCES projects(id)
        ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS narrative_unit_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    run_id INTEGER NOT NULL,
    project_id TEXT NOT NULL,

    chunk_database_id INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,

    text_hash TEXT NOT NULL,
    analysis_input_hash TEXT NOT NULL,

    status TEXT NOT NULL DEFAULT 'pending',
    cache_hit INTEGER NOT NULL DEFAULT 0
        CHECK (cache_hit IN (0, 1)),

    cache_source_unit_id INTEGER,

    provider TEXT NOT NULL,
    model TEXT NOT NULL,

    prompt_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,

    result_json TEXT,
    validated_result_json TEXT,
    error_message TEXT,

    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_started_at TEXT,
    last_finished_at TEXT,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (run_id)
        REFERENCES narrative_analysis_runs(id)
        ON DELETE CASCADE,

    FOREIGN KEY (project_id)
        REFERENCES projects(id)
        ON DELETE CASCADE,

    FOREIGN KEY (chunk_database_id)
        REFERENCES text_chunks(id)
        ON DELETE CASCADE,

    FOREIGN KEY (cache_source_unit_id)
        REFERENCES narrative_unit_analyses(id)
        ON DELETE SET NULL,

    UNIQUE (run_id, chunk_database_id)
);


CREATE TABLE IF NOT EXISTS narrative_layer_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    project_id TEXT NOT NULL,
    chunk_database_id INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,

    layer_name TEXT NOT NULL,
    input_hash TEXT NOT NULL,

    provider TEXT NOT NULL,
    model TEXT NOT NULL,

    prompt_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,

    result_json TEXT NOT NULL,
    validated_result_json TEXT NOT NULL,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (project_id)
        REFERENCES projects(id)
        ON DELETE CASCADE,

    FOREIGN KEY (chunk_database_id)
        REFERENCES text_chunks(id)
        ON DELETE CASCADE
);


CREATE INDEX IF NOT EXISTS idx_source_files_project_order
    ON source_files (
        project_id,
        file_order
    );


CREATE INDEX IF NOT EXISTS idx_chapters_source_order
    ON chapters (
        source_file_id,
        chapter_order
    );


CREATE INDEX IF NOT EXISTS idx_text_chunks_source_global_order
    ON text_chunks (
        source_file_id,
        global_order
    );


CREATE INDEX IF NOT EXISTS idx_text_chunks_chapter_order
    ON text_chunks (
        chapter_id,
        chunk_order_in_chapter
    );


CREATE INDEX IF NOT EXISTS idx_narrative_runs_project_created
    ON narrative_analysis_runs (
        project_id,
        created_at
    );


CREATE INDEX IF NOT EXISTS idx_narrative_units_run_chunk
    ON narrative_unit_analyses (
        run_id,
        chunk_database_id
    );


CREATE INDEX IF NOT EXISTS idx_narrative_units_project_hash
    ON narrative_unit_analyses (
        project_id,
        text_hash
    );


CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_analysis_per_project
    ON narrative_analysis_runs (
        project_id
    )
    WHERE status IN ('queued', 'running');
"""


def create_schema(
    connection: sqlite3.Connection,
) -> None:
    """创建/升级 Novel2Script 当前版本所需的数据库表、索引和迁移。"""

    connection.executescript(SCHEMA_SQL)

    _migrate_if_needed(connection)

    connection.execute(
        f"PRAGMA user_version = {DATABASE_SCHEMA_VERSION}"
    )


def _migrate_if_needed(
    connection: sqlite3.Connection,
) -> None:
    """按版本顺序执行所有尚未运行的迁移。"""
    current = _get_schema_version(connection)

    if current >= DATABASE_SCHEMA_VERSION:
        return

    if current < 2:
        _migrate_to_v2(connection)
    if current < 3:
        _migrate_to_v3(connection)
    if current < 4:
        _migrate_to_v4(connection)
    if current < 5:
        _migrate_to_v5(connection)


def _get_schema_version(
    connection: sqlite3.Connection,
) -> int:
    row = connection.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def _table_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> set[str]:
    rows = connection.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()
    return {row[1] for row in rows}


def _migrate_to_v2(
    connection: sqlite3.Connection,
) -> None:
    """v1 → v2: add analysis columns to narrative_unit_analyses."""
    existing_tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }

    if "narrative_unit_analyses" not in existing_tables:
        return

    columns = _table_columns(connection, "narrative_unit_analyses")

    if "analysis_input_hash" not in columns:
        connection.execute(
            """
            ALTER TABLE narrative_unit_analyses
            ADD COLUMN analysis_input_hash TEXT
            """
        )

    if "cache_hit" not in columns:
        connection.execute(
            """
            ALTER TABLE narrative_unit_analyses
            ADD COLUMN cache_hit INTEGER NOT NULL DEFAULT 0
            """
        )

    if "cache_source_unit_id" not in columns:
        connection.execute(
            """
            ALTER TABLE narrative_unit_analyses
            ADD COLUMN cache_source_unit_id INTEGER
            """
        )


def _migrate_to_v3(
    connection: sqlite3.Connection,
) -> None:
    """v2 → v3: add indexes for narrative_unit_analyses."""
    existing_indexes = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    }

    if "idx_narrative_units_cache_lookup" not in existing_indexes:
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_narrative_units_cache_lookup
            ON narrative_unit_analyses (
                project_id,
                analysis_input_hash,
                provider,
                model,
                prompt_version,
                schema_version,
                status
            )
            """
        )


def _migrate_to_v4(
    connection: sqlite3.Connection,
) -> None:
    """v3 → v4: deduplicate and add UNIQUE constraint on narrative_layer_cache."""
    existing_tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }

    if "narrative_layer_cache" not in existing_tables:
        return

    existing_indexes = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    }

    # Deduplicate BEFORE creating unique index.
    if "idx_narrative_layer_cache_unique" not in existing_indexes:
        connection.execute(
            """
            DELETE FROM narrative_layer_cache
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM narrative_layer_cache
                GROUP BY
                    project_id,
                    chunk_database_id,
                    layer_name,
                    input_hash,
                    provider,
                    model,
                    prompt_version,
                    schema_version
            )
            """
        )

        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_narrative_layer_cache_unique
            ON narrative_layer_cache (
                project_id,
                chunk_database_id,
                layer_name,
                input_hash,
                provider,
                model,
                prompt_version,
                schema_version
            )
            """
        )


def _add_column_if_missing(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_sql: str,
) -> None:
    if column_name in _table_columns(connection, table_name):
        return

    connection.execute(
        f"ALTER TABLE {table_name} ADD COLUMN {column_sql}"
    )


def _migrate_to_v5(
    connection: sqlite3.Connection,
) -> None:
    """v4 → v5: add async job progress fields and active-run lock."""

    existing_tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }

    if "narrative_analysis_runs" in existing_tables:
        run_columns = {
            "total_chunks": "total_chunks INTEGER NOT NULL DEFAULT 0",
            "processed_chunks": "processed_chunks INTEGER NOT NULL DEFAULT 0",
            "successful_chunks": "successful_chunks INTEGER NOT NULL DEFAULT 0",
            "partial_chunks": "partial_chunks INTEGER NOT NULL DEFAULT 0",
            "failed_chunks": "failed_chunks INTEGER NOT NULL DEFAULT 0",
            "cached_chunks": "cached_chunks INTEGER NOT NULL DEFAULT 0",
            "cached_layers": "cached_layers INTEGER NOT NULL DEFAULT 0",
            "current_chunk_id": "current_chunk_id TEXT",
            "request_json": "request_json TEXT",
            "started_at": "started_at TEXT",
            "finished_at": "finished_at TEXT",
            "heartbeat_at": "heartbeat_at TEXT",
        }

        for column_name, column_sql in run_columns.items():
            _add_column_if_missing(
                connection,
                "narrative_analysis_runs",
                column_name,
                column_sql,
            )

    if "narrative_unit_analyses" in existing_tables:
        unit_columns = {
            "attempt_count": "attempt_count INTEGER NOT NULL DEFAULT 0",
            "last_started_at": "last_started_at TEXT",
            "last_finished_at": "last_finished_at TEXT",
        }

        for column_name, column_sql in unit_columns.items():
            _add_column_if_missing(
                connection,
                "narrative_unit_analyses",
                column_name,
                column_sql,
            )

    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_analysis_per_project
        ON narrative_analysis_runs (
            project_id
        )
        WHERE status IN ('queued', 'running')
        """
    )
