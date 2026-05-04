import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class TourFeatureExtractor:
    """
    Extract features from tour data for content-based filtering.

    Features extracted:
    1. Destination (one-hot encoded)
    2. Price (normalized 0-1)
    3. Duration (extracted from "3 ngay 2 dem" format)
    4. Description TF-IDF (top features)
    """

    def __init__(self, max_tfidf_features: int = 50):
        self.destination_to_idx: Dict[str, int] = {}
        self.idx_to_destination: Dict[int, str] = {}
        self.price_scaler = MinMaxScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_tfidf_features)
        self.is_fitted = False
        self.num_destinations = 0
        self.feature_dim = 0

    def fit(self, tours: List[dict]):
        """Fit encoders on all tours."""
        if not tours:
            logger.warning("No tours provided for fitting")
            return

        # Extract unique destinations
        destinations = list(set(
            self._normalize_destination(t.get("destination", ""))
            for t in tours if t.get("destination")
        ))
        destinations.sort()

        self.destination_to_idx = {d: i for i, d in enumerate(destinations)}
        self.idx_to_destination = {i: d for d, i in self.destination_to_idx.items()}
        self.num_destinations = len(destinations)

        # Fit price scaler
        prices = [[t.get("priceAdult", 0)] for t in tours]
        if prices:
            self.price_scaler.fit(prices)

        # Fit TF-IDF on descriptions
        descriptions = [t.get("description", "") or "" for t in tours]
        if any(descriptions):
            self.tfidf_vectorizer.fit(descriptions)

        # Calculate feature dimension
        # destination_onehot + price + duration + tfidf
        tfidf_dim = len(self.tfidf_vectorizer.get_feature_names_out()) if hasattr(
            self.tfidf_vectorizer, 'vocabulary_') else 0
        self.feature_dim = self.num_destinations + 2 + tfidf_dim

        self.is_fitted = True
        logger.info(f"Feature extractor fitted: {self.num_destinations} destinations, "
                    f"{tfidf_dim} TF-IDF features, total dim: {self.feature_dim}")

    def _normalize_destination(self, destination: str) -> str:
        """Normalize destination string for matching."""
        if not destination:
            return ""
        # Remove diacritics and lowercase
        import unicodedata
        normalized = unicodedata.normalize("NFD", destination)
        normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
        return normalized.lower().strip()

    def _parse_duration(self, time_str: str) -> int:
        """
        Parse duration from Vietnamese format.
        Examples: "3 ngay 2 dem" -> 3, "4N3D" -> 4
        """
        if not time_str:
            return 1

        time_lower = time_str.lower()

        # Try "X ngay" pattern
        match = re.search(r"(\d+)\s*ng[aà]y", time_lower)
        if match:
            return int(match.group(1))

        # Try "XN" pattern (e.g., "4N3D")
        match = re.search(r"(\d+)\s*n", time_lower)
        if match:
            return int(match.group(1))

        # Try just finding a number
        match = re.search(r"(\d+)", time_str)
        if match:
            return int(match.group(1))

        return 1

    def extract_features(self, tour: dict) -> np.ndarray:
        """Extract feature vector for a single tour."""
        if not self.is_fitted:
            raise RuntimeError("Feature extractor not fitted. Call fit() first.")

        features = []

        # 1. Destination one-hot encoding
        dest = self._normalize_destination(tour.get("destination", ""))
        dest_onehot = np.zeros(self.num_destinations)
        if dest in self.destination_to_idx:
            dest_onehot[self.destination_to_idx[dest]] = 1.0
        features.extend(dest_onehot)

        # 2. Price normalized (0-1)
        price = tour.get("priceAdult", 0)
        try:
            price_norm = self.price_scaler.transform([[price]])[0][0]
        except Exception:
            price_norm = 0.5
        features.append(price_norm)

        # 3. Duration normalized (assume max 10 days)
        duration = self._parse_duration(tour.get("time", ""))
        duration_norm = min(duration / 10.0, 1.0)
        features.append(duration_norm)

        # 4. Description TF-IDF
        desc = tour.get("description", "") or ""
        try:
            tfidf = self.tfidf_vectorizer.transform([desc]).toarray()[0]
            features.extend(tfidf)
        except Exception:
            # If TF-IDF fails, pad with zeros
            tfidf_dim = len(self.tfidf_vectorizer.get_feature_names_out()) if hasattr(
                self.tfidf_vectorizer, 'vocabulary_') else 0
            features.extend([0.0] * tfidf_dim)

        return np.array(features, dtype=np.float32)

    def get_destination_from_idx(self, idx: int) -> str:
        """Get destination name from index."""
        return self.idx_to_destination.get(idx, "")

    def get_idx_from_destination(self, destination: str) -> int:
        """Get index from destination name."""
        normalized = self._normalize_destination(destination)
        return self.destination_to_idx.get(normalized, -1)


class TourSimilarityCalculator:
    """
    Pre-compute and store tour similarity matrix for fast lookups.
    Uses cosine similarity on tour feature vectors.
    """

    def __init__(self, feature_extractor: TourFeatureExtractor):
        self.feature_extractor = feature_extractor
        self.tour_features: Dict[str, np.ndarray] = {}
        self.similarity_matrix: Optional[np.ndarray] = None
        self.tour_ids: List[str] = []
        self.tour_id_to_idx: Dict[str, int] = {}

    def build_similarity_matrix(self, tours: List[dict]):
        """Pre-compute similarity matrix for all tours."""
        if not tours:
            logger.warning("No tours provided for similarity matrix")
            return

        self.tour_ids = [str(t["_id"]) for t in tours]
        self.tour_id_to_idx = {tid: i for i, tid in enumerate(self.tour_ids)}

        feature_matrix = []

        for tour in tours:
            tour_id = str(tour["_id"])
            try:
                features = self.feature_extractor.extract_features(tour)
                self.tour_features[tour_id] = features
                feature_matrix.append(features)
            except Exception as e:
                logger.error(f"Error extracting features for tour {tour_id}: {e}")
                # Use zero vector as fallback
                zero_features = np.zeros(self.feature_extractor.feature_dim)
                self.tour_features[tour_id] = zero_features
                feature_matrix.append(zero_features)

        feature_matrix = np.array(feature_matrix)

        # Compute cosine similarity matrix
        self.similarity_matrix = cosine_similarity(feature_matrix)

        logger.info(f"Similarity matrix built: {len(tours)} tours, shape {self.similarity_matrix.shape}")

    def get_similar_tours(self, tour_id: str, top_k: int = 4) -> List[Tuple[str, float]]:
        """
        Get top K most similar tours for a given tour.

        Returns:
            List of (tour_id, similarity_score) tuples, sorted by similarity descending.
        """
        if self.similarity_matrix is None:
            logger.warning("Similarity matrix not built")
            return []

        if tour_id not in self.tour_id_to_idx:
            logger.warning(f"Tour {tour_id} not found in similarity matrix")
            return []

        idx = self.tour_id_to_idx[tour_id]
        similarities = self.similarity_matrix[idx]

        # Get indices of top K+1 similar tours (including self)
        similar_indices = np.argsort(similarities)[::-1]

        result = []
        for i in similar_indices:
            if len(result) >= top_k:
                break
            # Skip self
            if i == idx:
                continue
            result.append((self.tour_ids[i], float(similarities[i])))

        return result

    def get_similarity(self, tour_id_1: str, tour_id_2: str) -> float:
        """Get similarity score between two tours."""
        if self.similarity_matrix is None:
            return 0.0

        if tour_id_1 not in self.tour_id_to_idx or tour_id_2 not in self.tour_id_to_idx:
            return 0.0

        idx1 = self.tour_id_to_idx[tour_id_1]
        idx2 = self.tour_id_to_idx[tour_id_2]

        return float(self.similarity_matrix[idx1, idx2])

    def get_tour_features(self, tour_id: str) -> Optional[np.ndarray]:
        """Get feature vector for a tour."""
        return self.tour_features.get(tour_id)
