import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from backend.storage.schema import create_schema


# backend/storage/database.py
# parents[0] -> backend/storage
# parents[1] -> backend
# parents[2] -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIRECTORY = PROJECT_ROOT / "data"

DEFAULT_DATABASE_PATH = (
    DATA_DIRECTORY / "novel2script.db"
)


DatabasePath = str | Path


def _prepare_database_location(
    database_path: DatabasePath | None,
) -> str | Path:
    """
    处理数据库地址。

    默认使用项目根目录下的 data/novel2script.db。
    同时保留 :memory: 支持，方便后续单元测试。
    """

    if database_path is None:
        return DEFAULT_DATABASE_PATH

    if str(database_path) == ":memory:":
        return ":memory:"

    return Path(database_path)


def connect_database(
    database_path: DatabasePath | None = None,
) -> sqlite3.Connection:
    """
    创建 SQLite 数据库连接。

    每次调用都会返回一个新的连接。调用方使用完毕后
    应提交或回滚事务，并关闭连接。
    """

    database_location = _prepare_database_location(
        database_path
    )

    is_memory_database = (
        database_location == ":memory:"
    )

    if not is_memory_database:
        path = Path(database_location)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        connection = sqlite3.connect(
            path,
            timeout=30.0,
        )
    else:
        connection = sqlite3.connect(
            ":memory:",
            timeout=30.0,
        )

    # 查询结果可通过字段名访问，例如 row["name"]。
    connection.row_factory = sqlite3.Row

    # SQLite 默认不会自动开启外键约束，
    # 因此每个连接都需要显式开启。
    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    # 数据库繁忙时等待最多 5 秒，避免立即抛出 locked 错误。
    connection.execute(
        "PRAGMA busy_timeout = 5000"
    )

    # WAL 模式更适合后续接口存在读写并发的场景。
    # 内存数据库不需要设置 WAL。
    if not is_memory_database:
        connection.execute(
            "PRAGMA journal_mode = WAL"
        )

    # 在可靠性和本地开发性能之间取得平衡。
    connection.execute(
        "PRAGMA synchronous = NORMAL"
    )

    return connection


@contextmanager
def database_session(
    database_path: DatabasePath | None = None,
) -> Iterator[sqlite3.Connection]:
    """
    提供带事务管理的数据库连接。

    正常结束：
        自动提交事务。

    发生异常：
        自动回滚事务，并继续抛出原异常。

    最后：
        自动关闭数据库连接。
    """

    connection = connect_database(
        database_path=database_path
    )

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(
    database_path: DatabasePath | None = None,
) -> None:
    """创建数据库文件、数据表和索引。"""

    with database_session(
        database_path=database_path
    ) as connection:
        create_schema(connection)