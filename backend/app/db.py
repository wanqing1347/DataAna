from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine


class Database:
    def __init__(self, url: str):
        self.engine: Engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=1800,
            future=True,
        )

    @contextmanager
    def connect(self):
        with self.engine.connect() as conn:
            yield conn

    @contextmanager
    def begin(self):
        with self.engine.begin() as conn:
            yield conn

    def dispose(self) -> None:
        self.engine.dispose()
