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

    async def get_all_tours(self) -> List[dict]:
        """Get all tours from database."""
        if self._tours_list_cache:
            return self._tours_list_cache

        tours_collection = get_collection("tbl_tours")
        cursor = tours_collection.find({})
        tours = await cursor.to_list(length=1000)

        # Convert ObjectId to string
        for tour in tours:
            tour["_id"] = str(tour["_id"])
            self._tours_cache[tour["_id"]] = tour

        self._tours_list_cache = tours
        return tours

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

        # Get all confirmed bookings
        cursor = bookings_collection.find({"bookingStatus": "c"})
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
                "bookingStatus": "c"
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
                "bookingStatus": "c"
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
                "bookingStatus": "c"
            })
            bookings = await cursor.to_list(length=1000)

            return list(set(str(b["userId"]) for b in bookings))
        except Exception as e:
            logger.error(f"Error getting users who booked {tour_id}: {e}")
            return []

    async def get_popular_tours(self, limit: int = 6) -> List[dict]:
        """Get most popular tours based on booking count and rating."""
        tours = await self.get_all_tours()

        # Calculate popularity score for each tour
        tour_scores = []
        for tour in tours:
            tour_id = tour["_id"]
            booking_count = await self.get_booking_count(tour_id)
            avg_rating = await self.get_avg_rating(tour_id)

            # Simple popularity score
            score = booking_count * 0.6 + avg_rating * 0.4
            tour_scores.append((tour, score))

        # Sort by score descending
        tour_scores.sort(key=lambda x: x[1], reverse=True)

        return [t[0] for t in tour_scores[:limit]]

    def clear_cache(self):
        """Clear internal caches."""
        self._tours_cache.clear()
        self._tours_list_cache = None
