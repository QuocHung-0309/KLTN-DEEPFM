from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Database:
    client: Optional[AsyncIOMotorClient] = None
    db = None


db = Database()


async def connect_to_mongo(uri: str, db_name: str = "travela"):
    """Connect to MongoDB using Motor async driver."""
    try:
        db.client = AsyncIOMotorClient(uri)
        db.db = db.client[db_name]

        # Verify connection
        await db.client.admin.command("ping")
        logger.info(f"Connected to MongoDB: {db_name}")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


async def close_mongo_connection():
    """Close MongoDB connection."""
    if db.client:
        db.client.close()
        logger.info("MongoDB connection closed")


def get_database():
    """Get database instance."""
    return db.db


def get_collection(name: str):
    """Get a specific collection."""
    return db.db[name]
