from typing import Dict, List, Set, Tuple, Optional
import logging

from app.services.collaborative import CollaborativeFilter
from app.services.content_based import ContentBasedFilter
from app.services.tour_service import TourService

logger = logging.getLogger(__name__)


class HybridRecommender:
    """
    Hybrid Recommendation System combining Collaborative Filtering and Content-Based.

    Hybrid Score Formula:
    score(u, t) = alpha * CF_score + beta * CB_score + gamma * popularity_score

    Dynamic Weight Adjustment based on user activity level:
    - Cold Start (0 bookings): alpha=0.0, beta=0.3, gamma=0.7
    - New User (1-2 bookings): alpha=0.2, beta=0.3, gamma=0.5
    - Regular (3-10 bookings): alpha=0.4, beta=0.4, gamma=0.2
    - Active (>10 bookings): alpha=0.5, beta=0.4, gamma=0.1

    Cold-Start Handling:
    - No history: Popular tours + diverse destinations
    - Some history: Content-based with popularity boost
    """

    # Weight configurations based on user activity level
    WEIGHTS = {
        "cold": {"cf": 0.0, "cb": 0.3, "pop": 0.7},       # 0 bookings
        "new": {"cf": 0.2, "cb": 0.3, "pop": 0.5},        # 1-2 bookings
        "regular": {"cf": 0.4, "cb": 0.4, "pop": 0.2},    # 3-10 bookings
        "active": {"cf": 0.5, "cb": 0.4, "pop": 0.1},     # >10 bookings
    }

    # Maximum tours per destination for diversity
    MAX_TOURS_PER_DESTINATION = 2

    def __init__(
        self,
        cf_filter: CollaborativeFilter,
        cb_filter: ContentBasedFilter,
        tour_service: TourService
    ):
        self.cf = cf_filter
        self.cb = cb_filter
        self.tour_service = tour_service
        self.popularity_scores: Dict[str, float] = {}
        self.tours_cache: Dict[str, dict] = {}

    async def initialize(self):
        """Initialize recommender with pre-computed popularity scores."""
        logger.info("Initializing hybrid recommender...")

        tours = await self.tour_service.get_all_tours()

        # Cache tours
        for tour in tours:
            self.tours_cache[str(tour["_id"])] = tour

        # Compute popularity scores
        await self._compute_popularity_scores(tours)

        logger.info(f"Hybrid recommender initialized with {len(tours)} tours")

    async def _compute_popularity_scores(self, tours: List[dict]):
        """
        Compute normalized popularity scores for all tours.

        Popularity = 0.6 * normalized_booking_count + 0.4 * normalized_rating
        """
        self.popularity_scores.clear()

        # Get booking counts and ratings
        tour_stats = []
        max_bookings = 1

        for tour in tours:
            tour_id = str(tour["_id"])
            booking_count = await self.tour_service.get_booking_count(tour_id)
            avg_rating = await self.tour_service.get_avg_rating(tour_id)

            tour_stats.append({
                "tour_id": tour_id,
                "booking_count": booking_count,
                "avg_rating": avg_rating
            })

            max_bookings = max(max_bookings, booking_count)

        # Normalize and compute popularity
        for stats in tour_stats:
            norm_bookings = stats["booking_count"] / max_bookings if max_bookings > 0 else 0
            norm_rating = stats["avg_rating"] / 5.0  # Rating 0-5 -> 0-1

            popularity = 0.6 * norm_bookings + 0.4 * norm_rating
            self.popularity_scores[stats["tour_id"]] = popularity

        logger.info(f"Computed popularity scores for {len(self.popularity_scores)} tours")

    def _get_user_level(self, booking_count: int) -> str:
        """Determine user activity level for weight selection."""
        if booking_count == 0:
            return "cold"
        elif booking_count <= 2:
            return "new"
        elif booking_count <= 10:
            return "regular"
        else:
            return "active"

    def _normalize_scores(self, scores: List[Tuple[str, float]]) -> Dict[str, float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return {}

        max_score = max(s[1] for s in scores) if scores else 1.0
        if max_score == 0:
            return {tour_id: 0.0 for tour_id, _ in scores}

        return {tour_id: score / max_score for tour_id, score in scores}

    def _apply_diversity(
        self,
        scored_tours: List[Tuple[str, float]],
        limit: int
    ) -> List[dict]:
        """
        Apply diversity constraint: max N tours per destination.
        Returns tour documents instead of IDs.
        """
        selected = []
        destination_counts: Dict[str, int] = {}

        for tour_id, score in scored_tours:
            if len(selected) >= limit:
                break

            tour = self.tours_cache.get(tour_id)
            if not tour:
                continue

            dest_slug = tour.get("destinationSlug", "").lower()

            # Check destination limit
            if dest_slug:
                current_count = destination_counts.get(dest_slug, 0)
                if current_count >= self.MAX_TOURS_PER_DESTINATION:
                    continue
                destination_counts[dest_slug] = current_count + 1

            selected.append(tour)

        return selected

    async def get_homepage_recommendations(
        self,
        user_id: str,
        limit: int = 6
    ) -> List[dict]:
        """
        Get personalized homepage recommendations.

        Strategy:
        1. Get user booking history
        2. Determine user level and weights
        3. Compute CF, CB, and popularity scores
        4. Combine with dynamic weights
        5. Apply diversity boost (different destinations)
        """
        # Get user history
        user_bookings = await self.tour_service.get_user_bookings(user_id)
        user_reviews = await self.tour_service.get_user_reviews(user_id)

        booked_tour_ids = set(b["tour_id"] for b in user_bookings if b.get("tour_id"))
        booking_count = len(booked_tour_ids)

        # Determine weights
        level = self._get_user_level(booking_count)
        weights = self.WEIGHTS[level]

        logger.info(f"User {user_id}: level={level}, bookings={booking_count}, weights={weights}")

        # Get all candidate tours
        all_tours = await self.tour_service.get_all_tours()
        all_tour_ids = [str(t["_id"]) for t in all_tours]

        # Build tours map
        tours_map = {str(t["_id"]): t for t in all_tours}

        # Compute scores from each method
        cf_scores: Dict[str, float] = {}
        cb_scores: Dict[str, float] = {}

        # Collaborative Filtering scores
        if weights["cf"] > 0 and self.cf.has_user_data(user_id):
            cf_recs = self.cf.get_recommendations(
                user_id, booked_tour_ids, all_tour_ids, top_k=limit * 3
            )
            cf_scores = self._normalize_scores(cf_recs)

        # Content-Based scores
        if weights["cb"] > 0 and booking_count > 0:
            # Build user profile from interactions
            interactions = []

            # Add bookings with weight 3.0
            for b in user_bookings:
                if b.get("tour_id"):
                    interactions.append({"tour_id": b["tour_id"], "weight": 3.0})

            # Add reviews with rating as weight
            for r in user_reviews:
                if r.get("tour_id"):
                    interactions.append({"tour_id": r["tour_id"], "weight": r.get("rating", 3.0)})

            self.cb.build_user_profile(user_id, interactions, tours_map)

            cb_recs = self.cb.get_recommendations(
                user_id, all_tours, booked_tour_ids, top_k=limit * 3
            )
            cb_scores = self._normalize_scores(cb_recs)

        # Combine scores
        combined = []

        for tour_id in all_tour_ids:
            if tour_id in booked_tour_ids:
                continue

            cf_s = cf_scores.get(tour_id, 0.0)
            cb_s = cb_scores.get(tour_id, 0.0)
            pop_s = self.popularity_scores.get(tour_id, 0.0)

            final_score = (
                weights["cf"] * cf_s +
                weights["cb"] * cb_s +
                weights["pop"] * pop_s
            )

            combined.append((tour_id, final_score))

        # Sort by score descending
        combined.sort(key=lambda x: x[1], reverse=True)

        # Apply diversity and return tour documents
        return self._apply_diversity(combined, limit)

    async def get_similar_tours(
        self,
        tour_id: str,
        limit: int = 4
    ) -> List[dict]:
        """
        Get content-based similar tours.
        Uses pre-computed similarity matrix.
        """
        similar = self.cb.get_similar_tours(tour_id, limit)

        tours = []
        for sim_tour_id, score in similar:
            tour = self.tours_cache.get(sim_tour_id)
            if tour:
                tours.append(tour)

        return tours

    async def get_post_booking_recommendations(
        self,
        tour_id: str,
        user_id: str,
        limit: int = 4
    ) -> List[dict]:
        """
        Post-booking recommendations combining similar tours and co-purchase patterns.

        Strategy:
        1. Get similar tours (content-based)
        2. Get co-purchased tours (what others who booked this also booked)
        3. Combine with weights: 0.6 similar + 0.4 co-purchase
        4. Exclude user's already booked tours
        """
        # Get user's booked tours to exclude
        user_bookings = await self.tour_service.get_user_bookings(user_id)
        exclude_ids = set(b["tour_id"] for b in user_bookings if b.get("tour_id"))
        exclude_ids.add(tour_id)

        # Get similar tours
        similar_tours = self.cb.get_similar_tours(tour_id, top_k=limit * 2)
        similar_scores = {t[0]: t[1] for t in similar_tours}

        # Get co-purchased tours
        co_purchased = self.cf.get_co_purchased_tours(
            tour_id, exclude_ids, top_k=limit * 2
        )

        # Normalize co-purchase counts to 0-1
        max_count = max((c[1] for c in co_purchased), default=1)
        co_purchase_scores = {
            t[0]: t[1] / max_count for t in co_purchased
        } if max_count > 0 else {}

        # Combine scores
        combined: Dict[str, float] = {}

        # Add similar tour scores (weight 0.6)
        for tour_id_s, score in similar_scores.items():
            if tour_id_s not in exclude_ids:
                combined[tour_id_s] = combined.get(tour_id_s, 0) + score * 0.6

        # Add co-purchase scores (weight 0.4)
        for tour_id_c, score in co_purchase_scores.items():
            if tour_id_c not in exclude_ids:
                combined[tour_id_c] = combined.get(tour_id_c, 0) + score * 0.4

        # Sort by combined score
        sorted_tours = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        # Get tour documents
        result = []
        for t_id, _ in sorted_tours[:limit]:
            tour = self.tours_cache.get(t_id)
            if tour:
                result.append(tour)

        # If not enough results, supplement with similar tours
        if len(result) < limit:
            for t_id, _ in similar_tours:
                if t_id not in exclude_ids and t_id not in [str(r["_id"]) for r in result]:
                    tour = self.tours_cache.get(t_id)
                    if tour:
                        result.append(tour)
                    if len(result) >= limit:
                        break

        return result

    async def refresh_popularity_scores(self):
        """Refresh popularity scores (can be called periodically)."""
        tours = await self.tour_service.get_all_tours()
        await self._compute_popularity_scores(tours)

    def get_popularity_score(self, tour_id: str) -> float:
        """Get popularity score for a tour."""
        return self.popularity_scores.get(tour_id, 0.0)
