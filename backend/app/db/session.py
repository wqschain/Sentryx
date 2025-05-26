from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from typing import AsyncGenerator
from ..core.config import settings

# Create async engine
engine = create_async_engine(
    settings.SQLITE_URL.replace("sqlite:///", "sqlite+aiosqlite:///"),
    echo=False,
    future=True,
    connect_args={"check_same_thread": False}
)

# Create async session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database sessions."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close() 