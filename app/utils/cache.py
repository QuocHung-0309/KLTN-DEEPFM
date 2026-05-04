from datetime import datetime, timedelta
from typing import Any, Optional, Dict
from functools import wraps
import hashlib
import logging

logger = logging.getLogger(__name__)


class SimpleCache:
    """
    Simple in-memory cache with TTL support.

    Cache Strategy:
    - Homepage recommendations: 2 minutes per user
    - Similar tours: 10 minutes per tour
    - Popularity scores: 1 hour
    - Tour features: Forever (invalidate on data refresh)
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._expiry: Dict[str, datetime] = {}

    def _make_key(self, prefix: str, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return None

        # Check expiry
        if key in self._expiry and datetime.now() > self._expiry[key]:
            del self._cache[key]
            del self._expiry[key]
            return None

        return self._cache[key]

    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        """
        Set value in cache with optional TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds (0 = no expiry)
        """
        self._cache[key] = value

        if ttl_seconds > 0:
            self._expiry[key] = datetime.now() + timedelta(seconds=ttl_seconds)
        elif key in self._expiry:
            del self._expiry[key]

    def delete(self, key: str):
        """Delete a specific key from cache."""
        if key in self._cache:
            del self._cache[key]
        if key in self._expiry:
            del self._expiry[key]

    def invalidate(self, pattern: Optional[str] = None):
        """
        Invalidate cache entries.

        Args:
            pattern: If provided, only invalidate keys containing this pattern.
                    If None, clear entire cache.
        """
        if pattern is None:
            self._cache.clear()
            self._expiry.clear()
            logger.info("Cache cleared")
        else:
            keys_to_delete = [k for k in self._cache if pattern in k]
            for k in keys_to_delete:
                del self._cache[k]
                if k in self._expiry:
                    del self._expiry[k]
            logger.info(f"Invalidated {len(keys_to_delete)} cache entries matching '{pattern}'")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        now = datetime.now()
        expired_count = sum(1 for exp in self._expiry.values() if now > exp)

        return {
            "total_entries": len(self._cache),
            "expired_entries": expired_count,
            "active_entries": len(self._cache) - expired_count
        }


# Global cache instance
cache = SimpleCache()


def cached(prefix: str, ttl: int = 300):
    """
    Decorator for caching async function results.

    Usage:
        @cached("homepage", ttl=120)
        async def get_homepage_recommendations(user_id: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            # Skip 'self' for methods
            cache_args = args[1:] if args and hasattr(args[0], '__class__') else args
            key = cache._make_key(prefix, *cache_args, **kwargs)

            # Check cache
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit: {prefix}")
                return cached_result

            # Execute function
            logger.debug(f"Cache miss: {prefix}")
            result = await func(*args, **kwargs)

            # Store in cache
            cache.set(key, result, ttl)

            return result

        return wrapper
    return decorator


def cached_sync(prefix: str, ttl: int = 300):
    """
    Decorator for caching sync function results.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_args = args[1:] if args and hasattr(args[0], '__class__') else args
            key = cache._make_key(prefix, *cache_args, **kwargs)

            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result

            result = func(*args, **kwargs)
            cache.set(key, result, ttl)

            return result

        return wrapper
    return decorator
