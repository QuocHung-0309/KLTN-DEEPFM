from typing import List, Dict, Optional, Set
from bson import ObjectId
import logging

from app.config.database import get_collection

logger = logging.getLogger(__name__)


class TourService:
    """Service for accessing tour, booking, and review data from MongoDB."""

    def __init__(self):
        self._tours_cache: Dict[str, dict] = {}
        self._tours_list_cache: Optional[List[dict]] = None

    async def get_all_tours(self, include_departures: bool = True) -> List[dict]:
        """Get all tours from database with upcoming departures."""
        if self._tours_list_cache:
            return self._tours_list_cache

        tours_collection = get_collection("tbl_tours")
        cursor = tours_collection.find({})
        tours = await cursor.to_list(length=1000)

        # Convert ObjectId to string
        for tour in tours:
            tour["_id"] = str(tour["_id"])
            self._tours_cache[tour["_id"]] = tour

        # Fetch upcoming departures for all tours
        if include_departures:
            await self._attach_upcoming_departures(tours)

        self._tours_list_cache = tours
        return tours

    async def _attach_upcoming_departures(self, tours: List[dict]):
        """Attach upcoming departures to each tour."""
        from datetime import datetime

        departures_collection = get_collection("tbl_tour_departures")
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Get all upcoming departures
        cursor = departures_collection.find({
            "startDate": {"$gte": today},
            "status": {"$in": ["pending", "confirmed"]}
        }).sort("startDate", 1)

        departures = await cursor.to_list(length=5000)

        # Group departures by tourId
        departures_by_tour: Dict[str, List[dict]] = {}
        for dep in departures:
            tour_id = str(dep.get("tourId", ""))
            if tour_id not in departures_by_tour:
                departures_by_tour[tour_id] = []

            departures_by_tour[tour_id].append({
                "_id": str(dep["_id"]),
                "startDate": dep.get("startDate").isoformat() if dep.get("startDate") else None,
                "endDate": dep.get("endDate").isoformat() if dep.get("endDate") else None,
                "max_guests": dep.get("max_guests", 0),
                "current_guests": dep.get("current_guests", 0),
                "priceAdult": dep.get("priceAdult", 0),
                "status": dep.get("status", "")
            })

        # Attach to tours
        for tour in tours:
            tour_id = tour["_id"]
            tour["upcomingDepartures"] = departures_by_tour.get(tour_id, [])[:5]  # Max 5 departures

    async def get_tour_by_id(self, tour_id: str) -> Optional[dict]:
        """Get a single tour by ID."""
        if tour_id in self._tours_cache:
            return self._tours_cache[tour_id]

        tours_collection = get_collection("tbl_tours")
        try:
            tour = await tours_collection.find_one({"_id": ObjectId(tour_id)})
            if tour:
                tour["_id"] = str(tour["_id"])
                self._tours_cache[tour_id] = tour
            return tour
        except Exception as e:
            logger.error(f"Error getting tour {tour_id}: {e}")
            return None

    async def get_all_bookings(self) -> List[dict]:
        """Get all confirmed bookings with tour departure info."""
        bookings_collection = get_collection("tbl_booking")
        departures_collection = get_collection("tbl_tour_departures")

        # Get all confirmed/completed bookings
        cursor = bookings_collection.find({"bookingStatus": {"$in": ["completed", "confirmed"]}})
        bookings = await cursor.to_list(length=10000)

        # Resolve tourId from tourDepartureId
        for booking in bookings:
            booking["_id"] = str(booking["_id"])
            booking["userId"] = str(booking["userId"])

            departure_id = booking.get("tourDepartureId")
            if departure_id:
                departure = await departures_collection.find_one(
                    {"_id": ObjectId(departure_id) if isinstance(departure_id, str) else departure_id}
                )
                if departure:
                    booking["tourId"] = str(departure.get("tourId", ""))
                else:
                    booking["tourId"] = ""

        return bookings

    async def get_all_reviews(self) -> List[dict]:
        """Get all reviews."""
        reviews_collection = get_collection("tbl_reviews")
        cursor = reviews_collection.find({})
        reviews = await cursor.to_list(length=10000)

        for review in reviews:
            review["_id"] = str(review["_id"])
            review["userId"] = str(review["userId"])
            review["tourId"] = str(review["tourId"])

        return reviews

    async def get_user_bookings(self, user_id: str) -> List[dict]:
        """Get all confirmed bookings for a specific user."""
        bookings_collection = get_collection("tbl_booking")
        departures_collection = get_collection("tbl_tour_departures")

        try:
            cursor = bookings_collection.find({
                "userId": ObjectId(user_id),
                "bookingStatus": {"$in": ["completed", "confirmed"]}
            })
            bookings = await cursor.to_list(length=100)

            result = []
            for booking in bookings:
                departure_id = booking.get("tourDepartureId")
                if departure_id:
                    departure = await departures_collection.find_one({"_id": departure_id})
                    if departure:
                        result.append({
                            "tour_id": str(departure.get("tourId", "")),
                            "booking_id": str(booking["_id"]),
                            "created_at": booking.get("createdAt")
                        })

            return result
        except Exception as e:
            logger.error(f"Error getting user bookings for {user_id}: {e}")
            return []

    async def get_user_reviews(self, user_id: str) -> List[dict]:
        """Get all reviews by a specific user."""
        reviews_collection = get_collection("tbl_reviews")

        try:
            cursor = reviews_collection.find({"userId": ObjectId(user_id)})
            reviews = await cursor.to_list(length=100)

            return [{
                "tour_id": str(r["tourId"]),
                "rating": r["rating"],
                "comment": r.get("comment", "")
            } for r in reviews]
        except Exception as e:
            logger.error(f"Error getting user reviews for {user_id}: {e}")
            return []

    async def get_booking_count(self, tour_id: str) -> int:
        """Get number of confirmed bookings for a tour."""
        bookings_collection = get_collection("tbl_booking")
        departures_collection = get_collection("tbl_tour_departures")

        try:
            # Find all departures for this tour
            cursor = departures_collection.find({"tourId": ObjectId(tour_id)})
            departures = await cursor.to_list(length=100)
            departure_ids = [d["_id"] for d in departures]

            if not departure_ids:
                return 0

            # Count confirmed bookings for these departures
            count = await bookings_collection.count_documents({
                "tourDepartureId": {"$in": departure_ids},
                "bookingStatus": {"$in": ["completed", "confirmed"]}
            })

            return count
        except Exception as e:
            logger.error(f"Error getting booking count for {tour_id}: {e}")
            return 0

    async def get_avg_rating(self, tour_id: str) -> float:
        """Get average rating for a tour."""
        reviews_collection = get_collection("tbl_reviews")

        try:
            pipeline = [
                {"$match": {"tourId": ObjectId(tour_id)}},
                {"$group": {"_id": None, "avg_rating": {"$avg": "$rating"}}}
            ]

            cursor = reviews_collection.aggregate(pipeline)
            result = await cursor.to_list(length=1)

            if result and result[0].get("avg_rating"):
                return result[0]["avg_rating"]
            return 0.0
        except Exception as e:
            logger.error(f"Error getting avg rating for {tour_id}: {e}")
            return 0.0

    async def get_users_who_booked(self, tour_id: str) -> List[str]:
        """Get list of user IDs who booked a specific tour."""
        bookings_collection = get_collection("tbl_booking")
        departures_collection = get_collection("tbl_tour_departures")

        try:
            # Find all departures for this tour
            cursor = departures_collection.find({"tourId": ObjectId(tour_id)})
            departures = await cursor.to_list(length=100)
            departure_ids = [d["_id"] for d in departures]

            if not departure_ids:
                return []

            # Find all users who booked these departures
            cursor = bookings_collection.find({
                "tourDepartureId": {"$in": departure_ids},
                "bookingStatus": {"$in": ["completed", "confirmed"]}
            })
            bookings = await cursor.to_list(length=1000)

            return list(set(str(b["userId"]) for b in bookings))
        except Exception as e:
            logger.error(f"Error getting users who booked {tour_id}: {e}")
            return []

    async def get_popular_tours(self, limit: int = 6) -> List[dict]:
        """Get most popular tours based on booking count and rating.

        Uses cached aggregation for performance.
        """
        from datetime import datetime, timedelta

        # Check cache
        cache_key = "_popular_tours_cache"
        cache_time_key = "_popular_tours_updated"

        if (hasattr(self, cache_key) and hasattr(self, cache_time_key) and
            getattr(self, cache_key) and getattr(self, cache_time_key) and
            datetime.now() - getattr(self, cache_time_key) < timedelta(minutes=5)):
            cached = getattr(self, cache_key)
            return cached[:limit]

        try:
            # Use aggregation pipeline to compute popularity scores efficiently
            # Step 1: Get all tours
            tours = await self.get_all_tours()
            tour_map = {str(t["_id"]): t for t in tours}

            # Step 2: Get booking counts per tour (via departures)
            bookings_collection = get_collection("tbl_booking")
            booking_pipeline = [
                {"$match": {"bookingStatus": {"$ne": "x"}}},
                {"$lookup": {
                    "from": "tbl_tour_departures",
                    "localField": "tourDepartureId",
                    "foreignField": "_id",
                    "as": "departure"
                }},
                {"$unwind": "$departure"},
                {"$group": {
                    "_id": "$departure.tourId",
                    "count": {"$sum": 1}
                }}
            ]
            booking_counts = {}
            async for doc in bookings_collection.aggregate(booking_pipeline):
                booking_counts[str(doc["_id"])] = doc["count"]

            # Step 3: Get average ratings per tour
            reviews_collection = get_collection("tbl_reviews")
            rating_pipeline = [
                {"$match": {"rating": {"$exists": True, "$gt": 0}}},
                {"$group": {
                    "_id": "$tourId",
                    "avgRating": {"$avg": "$rating"}
                }}
            ]
            ratings = {}
            async for doc in reviews_collection.aggregate(rating_pipeline):
                ratings[str(doc["_id"])] = doc["avgRating"]

            # Step 4: Calculate popularity scores
            tour_scores = []
            for tour_id, tour in tour_map.items():
                booking_count = booking_counts.get(tour_id, 0)
                avg_rating = ratings.get(tour_id, 4.0)  # Default rating

                # Popularity score
                score = booking_count * 0.6 + avg_rating * 0.4
                tour_scores.append((tour, score))

            # Sort by score descending
            tour_scores.sort(key=lambda x: x[1], reverse=True)

            # Cache result
            result = [t[0] for t in tour_scores]
            setattr(self, cache_key, result)
            setattr(self, cache_time_key, datetime.now())

            return result[:limit]

        except Exception as e:
            logger.error(f"Error getting popular tours: {e}")
            # Fallback to uncached version if aggregation fails
            tours = await self.get_all_tours()
            return tours[:limit]

    def clear_cache(self):
        """Clear internal caches."""
        self._tours_cache.clear()
        self._tours_list_cache = None
