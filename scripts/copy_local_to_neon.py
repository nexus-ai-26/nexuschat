"""Copy public database tables from LOCAL_DB_URI to DB_URI safely.

Both URLs are read from the environment and are never printed. Tables are
copied in foreign-key order, in batches, using ``ON CONFLICT DO NOTHING``.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine


SKIP_TABLES = {"alembic_version"}
BATCH_SIZE = 500


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def foreign_key_order(
    tables: Iterable[str], dependencies: Iterable[tuple[str, str]]
) -> list[str]:
    """Topologically order child/parent pairs as parent before child."""
    table_set = set(tables)
    parents = {
        child: {parent for parent in dependency_parents if parent in table_set}
        for child, dependency_parents in _group_dependencies(dependencies).items()
    }
    parents.update({table: set() for table in table_set if table not in parents})

    ordered: list[str] = []
    remaining = set(table_set)
    while remaining:
        ready = sorted(table for table in remaining if not (parents[table] & remaining))
        if not ready:
            # Preserve progress for legacy schemas with cyclic foreign keys.
            ordered.extend(sorted(remaining))
            break
        ordered.extend(ready)
        remaining.difference_update(ready)
    return ordered


def _group_dependencies(
    dependencies: Iterable[tuple[str, str]],
) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for child, parent in dependencies:
        grouped.setdefault(child, set()).add(parent)
    return grouped


def vector_literal(value: object) -> object:
    """Make a pgvector value portable when the driver returns a Python list."""
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(number) for number in value) + "]"
    return value


async def public_tables(connection: AsyncConnection) -> set[str]:
    result = await connection.execute(
        text(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """
        )
    )
    return {row[0] for row in result if row[0] not in SKIP_TABLES}


async def dependencies(
    connection: AsyncConnection,
) -> list[tuple[str, str]]:
    result = await connection.execute(
        text(
            """
            SELECT tc.table_name, ccu.table_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_schema = tc.constraint_schema
             AND ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_schema = 'public'
              AND tc.table_schema = 'public'
              AND tc.constraint_type = 'FOREIGN KEY'
            """
        )
    )
    return [(row[0], row[1]) for row in result]


async def columns(connection: AsyncConnection, table: str) -> list[tuple[str, str]]:
    result = await connection.execute(
        text(
            """
            SELECT column_name, udt_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = :table_name
              AND is_generated = 'NEVER'
            ORDER BY ordinal_position
            """
        ),
        {"table_name": table},
    )
    return [(row[0], row[1]) for row in result]


def insert_statement(table: str, table_columns: list[tuple[str, str]]):
    names = [name for name, _ in table_columns]
    quoted_names = ", ".join(quote_identifier(name) for name in names)
    placeholders = ", ".join(
        f"CAST(:{name} AS vector)" if udt_name == "vector" else f":{name}"
        for name, udt_name in table_columns
    )
    return text(
        f"INSERT INTO {quote_identifier('public')}.{quote_identifier(table)} "
        f"({quoted_names}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
    )


async def copy_table(
    source: AsyncConnection,
    target: AsyncConnection,
    table: str,
) -> int:
    table_columns = await columns(source, table)
    if not table_columns:
        return 0

    names = [name for name, _ in table_columns]
    vector_columns = {name for name, udt_name in table_columns if udt_name == "vector"}
    select_sql = text(
        f"SELECT {', '.join(quote_identifier(name) for name in names)} "
        f"FROM {quote_identifier('public')}.{quote_identifier(table)}"
    )
    insert_sql = insert_statement(table, table_columns)
    result = await source.execute(select_sql)
    copied = 0
    while rows := result.mappings().fetchmany(BATCH_SIZE):
        batch = []
        for row in rows:
            values = dict(row)
            for vector_column in vector_columns:
                values[vector_column] = vector_literal(values[vector_column])
            batch.append(values)
        inserted = await target.execute(insert_sql, batch)
        if inserted.rowcount and inserted.rowcount > 0:
            copied += inserted.rowcount
    return copied


async def copy_database(local_uri: str, target_uri: str) -> None:
    source_engine = create_async_engine(local_uri)
    target_engine = create_async_engine(target_uri)
    try:
        async with source_engine.connect() as source, target_engine.begin() as target:
            source_tables = await public_tables(source)
            target_tables = await public_tables(target)
            common_tables = source_tables & target_tables
            if not common_tables:
                print("no common public tables")
                return

            dependency_pairs = [
                pair
                for pair in await dependencies(source)
                if pair[0] in common_tables and pair[1] in common_tables
            ]
            for table in foreign_key_order(common_tables, dependency_pairs):
                copied = await copy_table(source, target, table)
                print(f"{table}: {copied}")
    finally:
        await source_engine.dispose()
        await target_engine.dispose()


def main() -> None:
    try:
        asyncio.run(
            copy_database(
                os.environ["LOCAL_DB_URI"],
                os.environ["DB_URI"],
            )
        )
    except Exception as error:
        # Do not echo exception text because database drivers can include URLs.
        print(f"copy failed: {type(error).__name__}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
