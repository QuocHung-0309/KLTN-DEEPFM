"""
DeepFM Recommender Service

Provides recommendation APIs using DeepFM model with fallback to popularity-based
recommendations for cold start scenarios.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from bson import ObjectId

from .deepfm_model import DeepFMRecommender
from .tour_service import TourService

logger = logging.getLogger(__name__)


class DeepFMRecommenderService:
    """
    Hybrid-primary + DeepFM re-rank Recommendation Service.

    Pipeline:
    1. HybridRecommender (CF + Content-based + Popularity) tạo candidate pool
    2. DeepFM re-score candidates để tinh chỉnh thứ hạng
    3. Blend score: 70% Hybrid + 30% DeepFM
    4. Diversity filter (max N tours per destination)

    Lý do dùng Hybrid làm primary:
    - Dataset nhỏ (71 users, 43 tours) → DeepFM dễ overfit
    - Hybrid đã xử lý tốt cold-start qua popularity + content-based
    - DeepFM bổ sung tín hiệu cá nhân hóa khi có đủ dữ liệu
    """

    def __init__(
        self,
        tour_service: TourService,
        model_path: str = "models/deepfm_libreco",
        max_per_destination: int = 2,
        hybrid_recommender=None,
    ):
        self.tour_service = tour_service
        self.model_path = model_path
        self.max_per_destination = max_per_destination
        self.hybrid_recommender = hybrid_recommender

        self.deepfm = DeepFMRecommender(model_path=model_path)
        self.db = None
        self.is_initialized = False

        # Cache for popular tours
        self._popular_tours_cache = None
        self._popular_tours_updated = None

    async def initialize(self, db) -> None:
        """Initialize the recommender with database connection."""
        self.db = db
        try:
            await self.deepfm.initialize(db)
            self.is_initialized = True
            logger.info("DeepFM Recommender Service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize DeepFM: {e}")
            self.is_initialized = False

    async def get_homepage_recommendations(
        self,
        user_id: Optional[str] = None,
        limit: int = 6
    ) -> List[Dict]:
        """
        Get personalized homepage recommendations.

        For logged-in users: Hybrid candidates + DeepFM re-rank
        For anonymous users: Popular tours
        """
        try:
            if user_id and self.is_initialized:
                return await self._get_deepfm_recommendations(user_id, limit)
            return await self._get_popular_tours(limit)
        except Exception as e:
            logger.error(f"Homepage recommendation error: {e}")
            return await self._get_popular_tours(limit)

    async def _get_deepfm_recommendations(
        self,
        user_id: str,
        limit: int
    ) -> List[Dict]:
        """
        Hybrid-primary + DeepFM re-rank pipeline.

        Bước 1: Hybrid tạo candidate pool (4× limit)
        Bước 2: DeepFM re-score → blend 70% Hybrid + 30% DeepFM
        Bước 3: Diversity filter → top-N
        """
        candidate_limit = limit * 4

        # ── Bước 1: Candidates từ Hybrid ─────────────────────────────────────
        candidates: List[Dict] = []
        if self.hybrid_recommender:
            try:
                candidates = await self.hybrid_recommender.get_homepage_recommendations(
                    user_id, candidate_limit
                )
            except Exception as e:
                logger.warning(f"Hybrid candidate generation failed: {e}")

        # Fallback nếu Hybrid không có kết quả
        if not candidates:
            all_tours = await self.tour_service.get_all_tours()
            candidates = [self._format_tour(t, 0.5) for t in all_tours]

        if not candidates:
            return []

        # ── Bước 2: DeepFM re-score ──────────────────────────────────────────
        if self.deepfm.is_trained and self.deepfm.model is not None:
            tour_ids = [c['_id'] for c in candidates]
            deepfm_scores = await self.deepfm.predict(user_id, tour_ids, self.db)
            deepfm_map = {tid: s for tid, s in deepfm_scores}

            # Normalize DeepFM scores về [0, 1]
            raw_scores = [deepfm_map.get(c['_id'], 0.5) for c in candidates]
            min_s, max_s = min(raw_scores), max(raw_scores)
            denom = (max_s - min_s) if max_s > min_s else 1.0

            for c in candidates:
                norm_deepfm = (deepfm_map.get(c['_id'], 0.5) - min_s) / denom
                hybrid_score = c.get('_score', 0.5)
                # Blend: Hybrid là primary (70%), DeepFM là tín hiệu bổ sung (30%)
                c['_score'] = 0.7 * hybrid_score + 0.3 * norm_deepfm

        # ── Bước 3: Sort + Diversity filter ──────────────────────────────────
        candidates.sort(key=lambda x: x.get('_score', 0), reverse=True)

        result = []
        dest_count: Dict[str, int] = {}
        for tour in candidates:
            dest = tour.get('destination', 'Unknown')
            if dest_count.get(dest, 0) >= self.max_per_destination:
                continue
            dest_count[dest] = dest_count.get(dest, 0) + 1
            result.append(tour)
            if len(result) >= limit:
                break

        return result

    async def _get_popular_tours(self, limit: int) -> List[Dict]:
        """Get popular tours based on bookings and ratings."""
        # Check cache
        if (self._popular_tours_cache and
            self._popular_tours_updated and
            datetime.now() - self._popular_tours_updated < timedelta(minutes=5)):
            return self._popular_tours_cache[:limit]

        # Calculate popularity scores
        pipeline = [
            {'$match': {'bookingStatus': {'$ne': 'x'}}},
            {'$lookup': {
                'from': 'tbl_tour_departures',
                'localField': 'tourDepartureId',
                'foreignField': '_id',
                'as': 'departure'
            }},
            {'$unwind': {'path': '$departure', 'preserveNullAndEmptyArrays': True}},
            {'$group': {
                '_id': '$departure.tourId',
                'booking_count': {'$sum': 1},
                'total_guests': {'$sum': {'$add': ['$numAdults', '$numChildren']}}
            }},
            {'$sort': {'booking_count': -1}},
            {'$limit': 20}
        ]

        booking_stats = await self.db.tbl_booking.aggregate(pipeline).to_list(20)

        # Get ratings
        rating_pipeline = [
            {'$group': {
                '_id': '$tourId',
                'avg_rating': {'$avg': '$rating'},
                'review_count': {'$sum': 1}
            }}
        ]
        rating_stats = await self.db.tbl_reviews.aggregate(rating_pipeline).to_list(100)
        ratings_map = {str(r['_id']): r for r in rating_stats}

        # Get tour details
        tour_ids = [b['_id'] for b in booking_stats if b['_id']]
        tours = await self.db.tbl_tours.find({
            '_id': {'$in': tour_ids}
        }).to_list(20)
        tour_map = {str(t['_id']): t for t in tours}

        # Calculate final scores
        results = []
        max_bookings = max((b['booking_count'] for b in booking_stats), default=1)

        for stat in booking_stats:
            tour_id = str(stat['_id']) if stat['_id'] else None
            if not tour_id or tour_id not in tour_map:
                continue

            tour = tour_map[tour_id]
            rating_info = ratings_map.get(tour_id, {'avg_rating': 4.0, 'review_count': 0})

            # Popularity score: 60% bookings + 40% rating
            booking_score = stat['booking_count'] / max_bookings
            rating_score = (rating_info['avg_rating'] or 4.0) / 5.0
            score = 0.6 * booking_score + 0.4 * rating_score

            results.append(self._format_tour(tour, score))

        # Sort by score
        results.sort(key=lambda x: x.get('_score', 0), reverse=True)

        # Apply diversity
        final_results = []
        dest_count = {}
        for tour in results:
            dest = tour.get('destination', 'Unknown')
            if dest_count.get(dest, 0) >= self.max_per_destination:
                continue
            dest_count[dest] = dest_count.get(dest, 0) + 1
            final_results.append(tour)

            if len(final_results) >= limit * 2:  # Cache more than needed
                break

        # Update cache
        self._popular_tours_cache = final_results
        self._popular_tours_updated = datetime.now()

        return final_results[:limit]

    async def get_similar_tours(
        self,
        tour_id: str,
        limit: int = 4
    ) -> List[Dict]:
        """Get similar tours based on content features."""
        try:
            # Get source tour
            source_tour = await self.db.tbl_tours.find_one({'_id': ObjectId(tour_id)})
            if not source_tour:
                return []

            # Get all tours
            all_tours = await self.db.tbl_tours.find({
                '_id': {'$ne': ObjectId(tour_id)}
            }).to_list(100)

            # Calculate similarity scores
            scored_tours = []
            for tour in all_tours:
                score = self._calculate_similarity(source_tour, tour)
                scored_tours.append((tour, score))

            # Sort by similarity
            scored_tours.sort(key=lambda x: x[1], reverse=True)

            # Apply diversity
            result = []
            dest_count = {}
            source_dest = source_tour.get('destination', '')

            for tour, score in scored_tours:
                dest = tour.get('destination', 'Unknown')

                # Skip same destination initially to increase diversity
                if dest == source_dest and len(result) < limit // 2:
                    if dest_count.get(dest, 0) >= 1:
                        continue

                if dest_count.get(dest, 0) >= self.max_per_destination:
                    continue

                dest_count[dest] = dest_count.get(dest, 0) + 1
                result.append(self._format_tour(tour, score))

                if len(result) >= limit:
                    break

            return result

        except Exception as e:
            logger.error(f"Similar tours error: {e}")
            return []

    def _calculate_similarity(self, source: Dict, target: Dict) -> float:
        """Calculate content-based similarity between tours."""
        score = 0.0

        # Same destination: +0.3
        if source.get('destination') == target.get('destination'):
            score += 0.3

        # Similar price range: +0.25
        source_price = source.get('priceAdult', 0)
        target_price = target.get('priceAdult', 0)
        if source_price > 0 and target_price > 0:
            price_ratio = min(source_price, target_price) / max(source_price, target_price)
            score += 0.25 * price_ratio

        # Similar duration: +0.2
        source_days = self._extract_days(source.get('time', ''))
        target_days = self._extract_days(target.get('time', ''))
        if source_days > 0 and target_days > 0:
            days_ratio = min(source_days, target_days) / max(source_days, target_days)
            score += 0.2 * days_ratio

        # Has images/itinerary: +0.15
        if target.get('images'):
            score += 0.1
        if target.get('itinerary'):
            score += 0.05

        # Boost newer tours: +0.1
        if target.get('createdAt'):
            days_old = (datetime.now() - target['createdAt']).days
            recency_score = max(0, 1 - days_old / 365)
            score += 0.1 * recency_score

        return score

    def _extract_days(self, time_str: str) -> int:
        """Extract number of days from time string."""
        import re
        if 'ngày' in time_str:
            match = re.search(r'(\d+)\s*ngày', time_str)
            if match:
                return int(match.group(1))
        return 0

    async def get_post_booking_recommendations(
        self,
        tour_id: str,
        user_id: Optional[str] = None,
        limit: int = 4
    ) -> List[Dict]:
        """
        Get recommendations after booking.

        Combines:
        - Similar tours (content-based)
        - Co-purchased tours (collaborative)
        - DeepFM predictions (if user is known)
        """
        try:
            results = []

            # 1. Get similar tours
            similar = await self.get_similar_tours(tour_id, limit=limit)
            results.extend(similar)

            # 2. Get co-purchased tours
            co_purchased = await self._get_co_purchased_tours(tour_id, limit=limit)
            for tour in co_purchased:
                if tour['_id'] not in [r['_id'] for r in results]:
                    results.append(tour)

            # 3. Boost with DeepFM scores for all logged-in users (including new users)
            if user_id and self.is_initialized and self.deepfm.is_trained:
                tour_ids = [r['_id'] for r in results]
                predictions = await self.deepfm.predict(user_id, tour_ids, self.db)
                pred_map = {p[0]: p[1] for p in predictions}

                for tour in results:
                    deepfm_score = pred_map.get(tour['_id'], 0.5)
                    current_score = tour.get('_score', 0.5)
                    # Blend scores: 60% original + 40% DeepFM
                    tour['_score'] = 0.6 * current_score + 0.4 * deepfm_score

                results.sort(key=lambda x: x.get('_score', 0), reverse=True)

            return results[:limit]

        except Exception as e:
            logger.error(f"Post-booking recommendation error: {e}")
            return await self.get_similar_tours(tour_id, limit)

    async def _get_co_purchased_tours(self, tour_id: str, limit: int) -> List[Dict]:
        """Get tours commonly booked by users who booked this tour."""
        try:
            # Get departure IDs for this tour
            departures = await self.db.tbl_tour_departures.find({
                'tourId': ObjectId(tour_id)
            }).to_list(100)
            dep_ids = [d['_id'] for d in departures]

            if not dep_ids:
                return []

            # Get users who booked this tour
            user_bookings = await self.db.tbl_booking.find({
                'tourDepartureId': {'$in': dep_ids},
                'bookingStatus': {'$ne': 'x'}
            }).to_list(1000)

            user_ids = list(set(b.get('userId') for b in user_bookings if b.get('userId')))

            if not user_ids:
                return []

            # Get other tours these users booked
            pipeline = [
                {'$match': {
                    'userId': {'$in': user_ids},
                    'tourDepartureId': {'$nin': dep_ids},
                    'bookingStatus': {'$ne': 'x'}
                }},
                {'$lookup': {
                    'from': 'tbl_tour_departures',
                    'localField': 'tourDepartureId',
                    'foreignField': '_id',
                    'as': 'departure'
                }},
                {'$unwind': '$departure'},
                {'$group': {
                    '_id': '$departure.tourId',
                    'count': {'$sum': 1}
                }},
                {'$sort': {'count': -1}},
                {'$limit': limit * 2}
            ]

            co_purchased = await self.db.tbl_booking.aggregate(pipeline).to_list(limit * 2)

            # Get tour details
            tour_ids = [c['_id'] for c in co_purchased if c['_id']]
            tours = await self.db.tbl_tours.find({
                '_id': {'$in': tour_ids}
            }).to_list(limit * 2)

            # Format and score
            results = []
            max_count = max((c['count'] for c in co_purchased), default=1)
            count_map = {str(c['_id']): c['count'] for c in co_purchased}

            for tour in tours:
                count = count_map.get(str(tour['_id']), 0)
                score = count / max_count
                results.append(self._format_tour(tour, score))

            results.sort(key=lambda x: x.get('_score', 0), reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Co-purchased tours error: {e}")
            return []

    def _format_tour(self, tour: Dict, score: float = 0.0) -> Dict:
        """Format tour for API response."""
        return {
            '_id': str(tour['_id']),
            'title': tour.get('title', ''),
            'destination': tour.get('destination', ''),
            'destinationSlug': tour.get('destinationSlug', ''),
            'priceAdult': tour.get('priceAdult', 0),
            'priceChild': tour.get('priceChild', 0),
            'time': tour.get('time', ''),
            'description': tour.get('description', '')[:200] if tour.get('description') else '',
            'images': tour.get('images', [])[:5],
            '_score': score
        }

    async def record_interaction(
        self,
        user_id: str,
        tour_id: str,
        interaction_type: str,
        value: float = 1.0
    ) -> None:
        """
        Record user interaction for online learning.

        interaction_type: 'booking', 'review', 'click', 'view'
        value: interaction strength (e.g., rating for review)
        """
        if not self.is_initialized:
            return

        # Convert interaction to label
        label_map = {
            'booking': 1.0,
            'review': value / 5.0,  # Normalize rating
            'click': 0.7,
            'view': 0.3
        }
        label = label_map.get(interaction_type, 0.5)

        # Update model
        await self.deepfm.online_update(self.db, {
            'user_id': user_id,
            'tour_id': tour_id,
            'label': label,
            'type': interaction_type
        })

    async def retrain(self, epochs: int = 10) -> Dict:
        """Retrain the model with all available data."""
        if self.db is None:
            return {"status": "error", "message": "Database not connected"}

        return await self.deepfm.train(self.db, epochs=epochs)

    def get_model_info(self) -> Dict:
        """Get model information and statistics."""
        return {
            "is_initialized": self.is_initialized,
            "is_trained": self.deepfm.is_trained if self.deepfm else False,
            "last_update": str(self.deepfm.last_update) if self.deepfm and self.deepfm.last_update else None,
            "num_users": len(self.deepfm.user_encoder) if self.deepfm else 0,
            "num_tours": len(self.deepfm.tour_encoder) if self.deepfm else 0,
            "num_destinations": len(self.deepfm.dest_encoder) if self.deepfm else 0,
            "model_path": self.model_path,
        }
