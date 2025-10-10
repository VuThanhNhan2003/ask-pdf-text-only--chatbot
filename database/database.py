"""
Database connection and session management
"""
import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import StaticPool
from .models import Base

# Database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./chatbot.db"  # Default to SQLite for development
)

# For PostgreSQL in production:
# DATABASE_URL = "postgresql://user:password@postgres:5432/chatbot_db"

# Engine configuration
if DATABASE_URL.startswith("sqlite"):
    # SQLite specific settings
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False  # Set True for SQL debugging
    )
else:
    # PostgreSQL/MySQL settings
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=False
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Thread-safe session
SessionScoped = scoped_session(SessionLocal)


def init_db():
    """Initialize database - create all tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized successfully")


def drop_db():
    """Drop all tables - USE WITH CAUTION"""
    Base.metadata.drop_all(bind=engine)
    print("🗑️ Database dropped")


@contextmanager
def get_db():
    """
    Get database session with automatic cleanup
    
    Usage:
        with get_db() as db:
            user = db.query(User).first()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_session():
    """
    Get database session (for dependency injection)
    Remember to close after use
    
    Usage:
        db = get_db_session()
        try:
            # do something
        finally:
            db.close()
    """
    return SessionLocal()