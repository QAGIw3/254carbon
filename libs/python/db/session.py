"""Database session management utilities."""

from __future__ import annotations

import contextlib
import logging
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


logger = logging.getLogger(__name__)


class SessionFactory:
    """Create SQLAlchemy sessions with consistent configuration."""

    def __init__(
        self,
        database_url: str,
        *,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_pre_ping: bool = True,
        future: bool = True,
    ) -> None:
        self.engine: Engine = create_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=pool_pre_ping,
            future=future,
        )
        self._session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            future=future,
        )

    @contextlib.contextmanager
    def session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:  # pragma: no cover - transaction rollback path
            logger.exception("Rolling back transaction due to exception")
            session.rollback()
            raise
        finally:
            session.close()


__all__ = ["SessionFactory"]

