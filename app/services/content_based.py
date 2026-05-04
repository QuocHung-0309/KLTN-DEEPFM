import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import logging

from app.services.feature_engineering import TourFeatureExtractor, TourSimilarityCalculator

logger = logging.getLogger(__name__)


class ContentBasedFilter:
    """
    Content-Based Filtering using user profile matching.

    User Profile Construction:
    profile(u) = sum[weight(u, t) * features(t)] / sum(weight)
                 (weighted average of interacted tour features)

    Prediction (Cosine Similarity):
    pred(u, t) = cos(profile(u), features(t))

    Cold-Start Strategy:
    - No history: Use popularity-based fallback (handled by HybridRecommender)
    - Little history: Build profile from available data
    """

    def __init__(
        self,
        feature_extractor: TourFeatureExtractor,
        similarity_calculator: TourSimilarityCalculator
    ):
        self.feature_extractor = feature_extractor
        self.similarity_calculator = similarity_calculator
        self.user_profiles: Dict[str, np.ndarray] = {}

    def build_user_profile(
        self,
        user_id: str,
        interactions: List[dict],
        tours_map: Dict[str, dict]
    ) -> Optional[np.ndarray]:
        """
        Build user preference profile from interaction history.

        Args:
            user_id: User identifier
            interactions: List of {tour_id, weight} dicts
            tours_map: Dictionary of tour_id -> tour_data

        Returns:
            User profile feature vector or None if insufficient data
        """
        if not interactions:
            return None

        weighted_features = []
        weights = []

        for interaction in interactions:
            tour_id = interaction.get("tour_id")
            weight = interaction.get("weight", 1.0)

            if not tour_id or tour_id not in tours_map:
                continue

            tour = tours_map[tour_id]

            try:
                features = self.feature_extractor.extract_features(tour)
                weighted_features.append(features * weight)
                weights.append(weight)
            except Exception as e:
                logger.warning(f"Error extracting features for tour {tour_id}: {e}")
                continue

        if not weighted_features:
            return None

        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return None

        profile = np.sum(weighted_features, axis=0) / total_weight

        # Normalize profile
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm

        self.user_profiles[user_id] = profile
        return profile

    def get_user_profile(self, user_id: str) -> Optional[np.ndarray]:
        """Get cached user profile."""
        return self.user_profiles.get(user_id)

    def predict(self, user_id: str, tour: dict) -> float:
        """
        Predict user preference score for a tour.

        Returns cosine similarity between user profile and tour features.
        Returns 0.0 if user has no profile.
        """
        if user_id not in self.user_profiles:
            return 0.0

        user_profile = self.user_profiles[user_id]

        try:
            tour_features = self.feature_extractor.extract_features(tour)
        except Exception as e:
            logger.warning(f"Error extracting tour features: {e}")
            return 0.0

        # Cosine similarity
        norm_user = np.linalg.norm(user_profile)
        norm_tour = np.linalg.norm(tour_features)

        if norm_user == 0 or norm_tour == 0:
            return 0.0

        return float(np.dot(user_profile, tour_features) / (norm_user * norm_tour))

    def predict_with_features(
        self,
        user_profile: np.ndarray,
        tour_features: np.ndarray
    ) -> float:
        """
        Predict score using pre-computed features.
        """
        norm_user = np.linalg.norm(user_profile)
        norm_tour = np.linalg.norm(tour_features)

        if norm_user == 0 or norm_tour == 0:
            return 0.0

        return float(np.dot(user_profile, tour_features) / (norm_user * norm_tour))

    def get_recommendations(
        self,
        user_id: str,
        candidate_tours: List[dict],
        exclude_tours: Set[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get content-based recommendations for a user.

        Args:
            user_id: Target user ID
            candidate_tours: List of all candidate tour documents
            exclude_tours: Set of tour IDs to exclude
            top_k: Number of recommendations to return

        Returns:
            List of (tour_id, score) tuples sorted by score descending
        """
        if user_id not in self.user_profiles:
            return []

        predictions = []

        for tour in candidate_tours:
            tour_id = str(tour.get("_id", ""))

            if tour_id in exclude_tours:
                continue

            score = self.predict(user_id, tour)
            if score > 0:
                predictions.append((tour_id, score))

        # Sort by score descending
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:top_k]

    def get_similar_tours(self, tour_id: str, top_k: int = 4) -> List[Tuple[str, float]]:
        """
        Get similar tours using pre-computed similarity matrix.
        Delegates to TourSimilarityCalculator.
        """
        return self.similarity_calculator.get_similar_tours(tour_id, top_k)

    def get_destination_based_recommendations(
        self,
        preferred_destinations: List[str],
        candidate_tours: List[dict],
        exclude_tours: Set[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get recommendations based on preferred destinations.
        Useful for cold-start users with some browsing history.
        """
        if not preferred_destinations:
            return []

        # Normalize destinations for matching
        import unicodedata

        def normalize(s):
            s = unicodedata.normalize("NFD", s)
            return "".join(c for c in s if unicodedata.category(c) != "Mn").lower().strip()

        preferred_set = set(normalize(d) for d in preferred_destinations)

        scored_tours = []

        for tour in candidate_tours:
            tour_id = str(tour.get("_id", ""))

            if tour_id in exclude_tours:
                continue

            dest = normalize(tour.get("destination", ""))
            dest_slug = normalize(tour.get("destinationSlug", ""))

            # Check if tour destination matches any preferred
            if dest in preferred_set or dest_slug in preferred_set:
                # Score based on position in preference list (earlier = higher)
                try:
                    idx = preferred_destinations.index(tour.get("destination", ""))
                    score = 1.0 - (idx * 0.1)  # First dest gets 1.0, second gets 0.9, etc.
                except ValueError:
                    score = 0.5

                scored_tours.append((tour_id, score))

        # Sort by score descending
        scored_tours.sort(key=lambda x: x[1], reverse=True)

        return scored_tours[:top_k]

    def clear_user_profiles(self):
        """Clear all cached user profiles."""
        self.user_profiles.clear()

    def has_user_profile(self, user_id: str) -> bool:
        """Check if user has a computed profile."""
        return user_id in self.user_profiles
