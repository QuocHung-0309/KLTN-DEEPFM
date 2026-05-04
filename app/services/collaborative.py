import numpy as np
from typing import Dict, List, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class CollaborativeFilter:
    """
    User-Based Collaborative Filtering using implicit feedback.

    User-Item Matrix Construction:
    - rating(u, t) = booking_weight + review_weight
    - booking_weight = 3.0 (implicit positive signal)
    - review_weight = review.rating (1-5, explicit signal)

    User Similarity (Cosine):
    sim(u1, u2) = (R_u1 . R_u2) / (||R_u1|| * ||R_u2||)

    Prediction:
    pred(u, t) = sum[sim(u, v) * rating(v, t)] / sum|sim(u, v)|
                 for all v in neighbors(u)
    """

    BOOKING_WEIGHT = 3.0  # Weight for completed booking
    MIN_NEIGHBORS = 2     # Minimum similar users for prediction
    MAX_NEIGHBORS = 20    # Maximum neighbors to consider
    MIN_SIMILARITY = 0.1  # Minimum similarity threshold

    def __init__(self):
        self.user_ratings: Dict[str, Dict[str, float]] = {}  # user_id -> {tour_id: rating}
        self.tour_users: Dict[str, Set[str]] = {}            # tour_id -> set(user_ids)
        self.user_similarity_cache: Dict[Tuple[str, str], float] = {}
        self.all_tour_ids: Set[str] = set()

    async def build_model(self, bookings: List[dict], reviews: List[dict]):
        """Build user-item rating matrix from bookings and reviews."""
        logger.info(f"Building CF model from {len(bookings)} bookings and {len(reviews)} reviews")

        self.user_ratings.clear()
        self.tour_users.clear()
        self.user_similarity_cache.clear()
        self.all_tour_ids.clear()

        # Process bookings (implicit feedback = 3.0 weight)
        for booking in bookings:
            user_id = str(booking.get("userId", ""))
            tour_id = str(booking.get("tourId", ""))

            if not user_id or not tour_id:
                continue

            if user_id not in self.user_ratings:
                self.user_ratings[user_id] = {}

            # Set booking weight
            self.user_ratings[user_id][tour_id] = self.BOOKING_WEIGHT

            if tour_id not in self.tour_users:
                self.tour_users[tour_id] = set()
            self.tour_users[tour_id].add(user_id)

            self.all_tour_ids.add(tour_id)

        # Process reviews (explicit feedback = rating 1-5)
        for review in reviews:
            user_id = str(review.get("userId", ""))
            tour_id = str(review.get("tourId", ""))
            rating = review.get("rating", 0)

            if not user_id or not tour_id or not rating:
                continue

            if user_id not in self.user_ratings:
                self.user_ratings[user_id] = {}

            # Use max of booking weight and review rating
            current = self.user_ratings[user_id].get(tour_id, 0)
            self.user_ratings[user_id][tour_id] = max(current, rating)

            if tour_id not in self.tour_users:
                self.tour_users[tour_id] = set()
            self.tour_users[tour_id].add(user_id)

            self.all_tour_ids.add(tour_id)

        logger.info(f"CF model built: {len(self.user_ratings)} users, {len(self.all_tour_ids)} tours")

    def _compute_user_similarity(self, u1: str, u2: str) -> float:
        """Compute cosine similarity between two users."""
        # Check cache
        cache_key = (min(u1, u2), max(u1, u2))
        if cache_key in self.user_similarity_cache:
            return self.user_similarity_cache[cache_key]

        ratings1 = self.user_ratings.get(u1, {})
        ratings2 = self.user_ratings.get(u2, {})

        # Find common tours
        common_tours = set(ratings1.keys()) & set(ratings2.keys())

        if not common_tours:
            self.user_similarity_cache[cache_key] = 0.0
            return 0.0

        # Build vectors for common tours
        vec1 = np.array([ratings1[t] for t in common_tours])
        vec2 = np.array([ratings2[t] for t in common_tours])

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            self.user_similarity_cache[cache_key] = 0.0
            return 0.0

        similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))

        # Cache the result
        self.user_similarity_cache[cache_key] = similarity

        return similarity

    def _get_neighbors(self, user_id: str) -> List[Tuple[str, float]]:
        """Get similar users sorted by similarity descending."""
        if user_id not in self.user_ratings:
            return []

        neighbors = []

        for other_user in self.user_ratings.keys():
            if other_user == user_id:
                continue

            sim = self._compute_user_similarity(user_id, other_user)
            if sim >= self.MIN_SIMILARITY:
                neighbors.append((other_user, sim))

        # Sort by similarity descending
        neighbors.sort(key=lambda x: x[1], reverse=True)

        return neighbors[:self.MAX_NEIGHBORS]

    def predict(self, user_id: str, tour_id: str) -> float:
        """
        Predict rating for a user-tour pair.

        Returns 0.0 if:
        - User has already interacted with this tour
        - Not enough neighbors to make prediction
        """
        # Check if user already interacted with this tour
        if user_id in self.user_ratings and tour_id in self.user_ratings[user_id]:
            return 0.0

        neighbors = self._get_neighbors(user_id)

        if len(neighbors) < self.MIN_NEIGHBORS:
            return 0.0

        numerator = 0.0
        denominator = 0.0

        for neighbor_id, similarity in neighbors:
            if tour_id in self.user_ratings.get(neighbor_id, {}):
                rating = self.user_ratings[neighbor_id][tour_id]
                numerator += similarity * rating
                denominator += abs(similarity)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def get_recommendations(
        self,
        user_id: str,
        exclude_tours: Set[str],
        candidate_tour_ids: List[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top K tour recommendations for a user.

        Args:
            user_id: Target user ID
            exclude_tours: Set of tour IDs to exclude (already booked)
            candidate_tour_ids: List of all available tour IDs
            top_k: Number of recommendations to return

        Returns:
            List of (tour_id, predicted_score) tuples
        """
        if user_id not in self.user_ratings:
            # Cold start user - no CF predictions possible
            return []

        predictions = []

        for tour_id in candidate_tour_ids:
            if tour_id in exclude_tours:
                continue

            score = self.predict(user_id, tour_id)
            if score > 0:
                predictions.append((tour_id, score))

        # Sort by score descending
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:top_k]

    def get_co_purchased_tours(
        self,
        tour_id: str,
        exclude_tours: Set[str],
        top_k: int = 10
    ) -> List[Tuple[str, int]]:
        """
        Get tours that are commonly booked by users who also booked this tour.

        Returns:
            List of (tour_id, co_purchase_count) tuples
        """
        if tour_id not in self.tour_users:
            return []

        users_who_booked = self.tour_users[tour_id]

        co_purchase_count: Dict[str, int] = {}

        for user_id in users_who_booked:
            user_tours = self.user_ratings.get(user_id, {}).keys()
            for other_tour in user_tours:
                if other_tour == tour_id or other_tour in exclude_tours:
                    continue
                co_purchase_count[other_tour] = co_purchase_count.get(other_tour, 0) + 1

        # Sort by count descending
        sorted_tours = sorted(
            co_purchase_count.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_tours[:top_k]

    def has_user_data(self, user_id: str) -> bool:
        """Check if user has any interaction data."""
        return user_id in self.user_ratings and len(self.user_ratings[user_id]) > 0

    def get_user_booking_count(self, user_id: str) -> int:
        """Get number of tours user has interacted with."""
        return len(self.user_ratings.get(user_id, {}))
