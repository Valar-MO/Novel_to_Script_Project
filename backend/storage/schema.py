import sqlite3


DATABASE_SCHEMA_VERSION = 1


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
"""


def create_schema(
    connection: sqlite3.Connection,
) -> None:
    """创建 Novel2Script 当前版本所需的数据库表和索引。"""

    connection.executescript(SCHEMA_SQL)

    connection.execute(
        f"PRAGMA user_version = {DATABASE_SCHEMA_VERSION}"
    )