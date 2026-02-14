from __future__ import annotations 
# for Python 3.7-3.9 compatibility with type hints - python delays type evals
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DB_URL = os.getenv(
    "DB_URL",
    "postgresql+psycopg2://grc_user:grc_pass@localhost:5432/grc_db"
    # dialect+driver://username:password@host:port/database

)
# If an environment variable called DB_URL exists, use that. Otherwise use this default string

# creates database engine which manages connections to the PostgreSQL database using SQLAlchemy.
# pre_ping: When connections sit idle, PostgreSQL may drop them.
engine = create_engine(DB_URL, pool_pre_ping=True)

# creates session factory - transaction work space
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
# autocommit=False means we have to explicitly call commit() to save changes
# autoflush=False means changes won't be sent to the database until we call commit()

# creates base class for all ORM models
Base = declarative_base()
