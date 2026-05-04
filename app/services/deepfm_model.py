"""
DeepFM Model Implementation using TensorFlow/Keras

DeepFM = Linear + FM + DNN

Architecture:
- Linear Part: First-order feature importance (w₀ + Σ wᵢxᵢ)
- FM Part: Second-order feature interactions (Σᵢ Σⱼ <vᵢ, vⱼ> xᵢxⱼ)
- DNN Part: High-order feature interactions

Fields (8 total):
- user_id (sparse)
- tour_id (sparse)
- destination_id (sparse)
- price_bucket (sparse) - giá tour chia bucket
- duration_bucket (sparse) - số ngày chia bucket
- has_images (sparse) - 0/1
- has_itinerary (sparse) - 0/1
- season (sparse) - mùa du lịch
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from typing import Dict, List, Tuple, Optional
import logging
import os
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)

# Constants
NUM_PRICE_BUCKETS = 10
NUM_DURATION_BUCKETS = 7
NUM_SEASONS = 4


class LinearLayer(layers.Layer):
    """Linear part: w₀ + Σ wᵢxᵢ for first-order feature importance."""

    def __init__(self, num_features: int, l2_reg: float = 1e-5, **kwargs):
        super(LinearLayer, self).__init__(**kwargs)
        self.num_features = num_features
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.bias = self.add_weight(
            name='linear_bias',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )
        self.built = True

    def call(self, field_embeddings_list):
        """
        inputs: list of (batch, 1) tensors - one weight per field
        """
        # Sum all linear terms + bias
        linear_out = self.bias
        for emb in field_embeddings_list:
            # Each embedding is (batch, emb_dim), take first dim as linear weight
            linear_out = linear_out + tf.reduce_sum(emb[:, :1], axis=1, keepdims=True)
        return linear_out


class FMLayer(layers.Layer):
    """
    Factorization Machine Layer for second-order feature interactions.

    FM: 0.5 * Σₖ[(Σᵢ vᵢₖxᵢ)² - Σᵢ vᵢₖ²xᵢ²]

    This captures all pairwise interactions between fields:
    - user × destination
    - user × price_bucket
    - destination × duration
    - price × duration
    - etc.
    """

    def __init__(self, **kwargs):
        super(FMLayer, self).__init__(**kwargs)

    def call(self, field_embeddings):
        """
        inputs shape: (batch_size, num_fields, embedding_dim)
        Each field has its own embedding vector.
        """
        # Sum of embeddings: (batch, emb_dim)
        sum_of_emb = tf.reduce_sum(field_embeddings, axis=1)
        # Square of sum: (batch, emb_dim)
        square_of_sum = tf.square(sum_of_emb)

        # Sum of squares: (batch, emb_dim)
        square_of_emb = tf.square(field_embeddings)
        sum_of_square = tf.reduce_sum(square_of_emb, axis=1)

        # FM output: (batch, 1)
        fm_output = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=1, keepdims=True)
        return fm_output


class DNNLayer(layers.Layer):
    """Deep Neural Network for high-order feature interactions."""

    def __init__(
        self,
        hidden_units: List[int] = [256, 128, 64],
        activation: str = 'relu',
        dropout_rate: float = 0.2,
        l2_reg: float = 1e-5,
        **kwargs
    ):
        super(DNNLayer, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate

        self.dense_layers = []
        self.bn_layers = []
        self.dropout_layers = []

        for units in hidden_units:
            self.dense_layers.append(
                layers.Dense(
                    units,
                    activation=activation,
                    kernel_regularizer=regularizers.l2(l2_reg)
                )
            )
            self.bn_layers.append(layers.BatchNormalization())
            self.dropout_layers.append(layers.Dropout(dropout_rate))

    def call(self, inputs, training=False):
        x = inputs
        for dense, bn, dropout in zip(self.dense_layers, self.bn_layers, self.dropout_layers):
            x = dense(x)
            x = bn(x, training=training)
            x = dropout(x, training=training)
        return x


class DeepFMModel(Model):
    """
    DeepFM Model for Tour Recommendation.

    y = σ(y_linear + y_FM + y_DNN)

    Fields (8 sparse features):
    1. user_id - người dùng
    2. tour_id - tour
    3. destination_id - điểm đến
    4. price_bucket - bucket giá (0-9)
    5. duration_bucket - bucket số ngày (0-6)
    6. has_images - có ảnh không (0/1)
    7. has_itinerary - có lịch trình không (0/1)
    8. season - mùa (0-3: xuân/hạ/thu/đông)
    """

    def __init__(
        self,
        num_users: int,
        num_tours: int,
        num_destinations: int,
        embedding_dim: int = 16,
        dnn_hidden_units: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3,
        l2_reg: float = 1e-4,
        **kwargs
    ):
        super(DeepFMModel, self).__init__(**kwargs)

        self.num_users = num_users
        self.num_tours = num_tours
        self.num_destinations = num_destinations
        self.embedding_dim = embedding_dim
        self.num_fields = 8  # Total number of fields

        # Embedding layers for each field
        # Field 1: user_id
        self.user_embedding = layers.Embedding(
            num_users + 1, embedding_dim,
            embeddings_regularizer=regularizers.l2(l2_reg),
            name='user_embedding'
        )

        # Field 2: tour_id
        self.tour_embedding = layers.Embedding(
            num_tours + 1, embedding_dim,
            embeddings_regularizer=regularizers.l2(l2_reg),
            name='tour_embedding'
        )

        # Field 3: destination_id
        self.dest_embedding = layers.Embedding(
            num_destinations + 1, embedding_dim,
            embeddings_regularizer=regularizers.l2(l2_reg),
            name='destination_embedding'
        )

        # Field 4: price_bucket (10 buckets)
        self.price_embedding = layers.Embedding(
            NUM_PRICE_BUCKETS + 1, embedding_dim,
            embeddings_regularizer=regularizers.l2(l2_reg),
            name='price_embedding'
        )

        # Field 5: duration_bucket (7 buckets: 1-7+ days)
        self.duration_embedding = layers.Embedding(
            NUM_DURATION_BUCKETS + 1, embedding_dim,
            embeddings_regularizer=regularizers.l2(l2_reg),
            name='duration_embedding'
        )

        # Field 6: has_images (0/1)
        self.images_embedding = layers.Embedding(
            2, embedding_dim,
            embeddings_regularizer=regularizers.l2(l2_reg),
            name='images_embedding'
        )

        # Field 7: has_itinerary (0/1)
        self.itinerary_embedding = layers.Embedding(
            2, embedding_dim,
            embeddings_regularizer=regularizers.l2(l2_reg),
            name='itinerary_embedding'
        )

        # Field 8: season (0-3)
        self.season_embedding = layers.Embedding(
            NUM_SEASONS + 1, embedding_dim,
            embeddings_regularizer=regularizers.l2(l2_reg),
            name='season_embedding'
        )

        # Linear Layer (first-order)
        self.linear_layer = LinearLayer(self.num_fields, l2_reg=l2_reg, name='linear')

        # FM Layer (second-order)
        self.fm_layer = FMLayer(name='fm_layer')

        # DNN Layer (high-order)
        self.dnn_layer = DNNLayer(
            hidden_units=dnn_hidden_units,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            name='dnn_layer'
        )

        # Output layers
        self.dnn_output = layers.Dense(1, name='dnn_output')
        self.final_output = layers.Dense(1, activation='sigmoid', name='output')

        # Flatten layer
        self.flatten = layers.Flatten()

    def call(self, inputs, training=False):
        """
        Forward pass.

        inputs: Dict with keys:
            - user_id: (batch_size,)
            - tour_id: (batch_size,)
            - destination_id: (batch_size,)
            - price_bucket: (batch_size,)
            - duration_bucket: (batch_size,)
            - has_images: (batch_size,)
            - has_itinerary: (batch_size,)
            - season: (batch_size,)
        """
        # Get embeddings for all 8 fields
        user_emb = self.user_embedding(inputs['user_id'])           # (batch, emb_dim)
        tour_emb = self.tour_embedding(inputs['tour_id'])           # (batch, emb_dim)
        dest_emb = self.dest_embedding(inputs['destination_id'])    # (batch, emb_dim)
        price_emb = self.price_embedding(inputs['price_bucket'])    # (batch, emb_dim)
        dur_emb = self.duration_embedding(inputs['duration_bucket'])# (batch, emb_dim)
        img_emb = self.images_embedding(inputs['has_images'])       # (batch, emb_dim)
        itin_emb = self.itinerary_embedding(inputs['has_itinerary'])# (batch, emb_dim)
        season_emb = self.season_embedding(inputs['season'])        # (batch, emb_dim)

        # List of all field embeddings for linear layer
        field_emb_list = [user_emb, tour_emb, dest_emb, price_emb,
                          dur_emb, img_emb, itin_emb, season_emb]

        # Stack embeddings for FM: (batch, 8, emb_dim)
        stacked_emb = tf.stack(field_emb_list, axis=1)

        # === Linear Part (first-order) ===
        linear_output = self.linear_layer(field_emb_list)  # (batch, 1)

        # === FM Part (second-order) ===
        fm_output = self.fm_layer(stacked_emb)  # (batch, 1)

        # === DNN Part (high-order) ===
        dnn_input = self.flatten(stacked_emb)  # (batch, 8 * emb_dim)
        dnn_hidden = self.dnn_layer(dnn_input, training=training)  # (batch, 64)
        dnn_output = self.dnn_output(dnn_hidden)  # (batch, 1)

        # === Combine: Linear + FM + DNN ===
        combined = linear_output + fm_output + dnn_output
        output = self.final_output(combined)

        return output

    def get_config(self):
        return {
            'num_users': self.num_users,
            'num_tours': self.num_tours,
            'num_destinations': self.num_destinations,
            'embedding_dim': self.embedding_dim,
        }


class DeepFMRecommender:
    """
    DeepFM-based Tour Recommender with Online Learning support.
    """

    def __init__(
        self,
        model_path: str = "models/deepfm",
        embedding_dim: int = 16,
        learning_rate: float = 0.001,
        batch_size: int = 64,
    ):
        self.model_path = model_path
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.model: Optional[DeepFMModel] = None
        self.user_encoder: Dict[str, int] = {}
        self.tour_encoder: Dict[str, int] = {}
        self.dest_encoder: Dict[str, int] = {}

        self.is_trained = False
        self.last_update = None

        # Feature statistics for bucketing
        self.price_boundaries = []  # Will be computed from data

    async def initialize(self, db) -> None:
        """Initialize model with data from database."""
        logger.info("Initializing DeepFM Recommender...")

        # Load existing model or create new
        if self._load_model():
            logger.info("Loaded existing DeepFM model")
        else:
            await self._build_encoders(db)
            await self._compute_price_boundaries(db)
            self._create_model()
            logger.info("Created new DeepFM model")

        # Initial training if not trained
        if not self.is_trained:
            await self.train(db)

    async def _build_encoders(self, db) -> None:
        """Build ID encoders for sparse features."""
        # Users
        users = await db.tbl_users.find({}, {'_id': 1}).to_list(10000)
        self.user_encoder = {str(u['_id']): i + 1 for i, u in enumerate(users)}

        # Tours
        tours = await db.tbl_tours.find({}, {'_id': 1, 'destination': 1}).to_list(1000)
        self.tour_encoder = {str(t['_id']): i + 1 for i, t in enumerate(tours)}

        # Destinations
        destinations = set(t.get('destination', 'Unknown') for t in tours)
        self.dest_encoder = {d: i + 1 for i, d in enumerate(destinations)}

        logger.info(f"Encoders built: {len(self.user_encoder)} users, "
                   f"{len(self.tour_encoder)} tours, {len(self.dest_encoder)} destinations")

    async def _compute_price_boundaries(self, db) -> None:
        """Compute price bucket boundaries using quantiles."""
        tours = await db.tbl_tours.find({}).to_list(1000)
        prices = [t.get('priceAdult', 0) for t in tours if t.get('priceAdult', 0) > 0]

        if prices:
            # Create 10 buckets using percentiles
            self.price_boundaries = [
                np.percentile(prices, p) for p in range(10, 100, 10)
            ]
        else:
            # Default boundaries (in VND)
            self.price_boundaries = [
                1000000, 2000000, 3000000, 4000000, 5000000,
                6000000, 8000000, 10000000, 15000000
            ]

        logger.info(f"Price boundaries: {self.price_boundaries}")

    def _get_price_bucket(self, price: float) -> int:
        """Convert price to bucket index (0-9)."""
        for i, boundary in enumerate(self.price_boundaries):
            if price <= boundary:
                return i
        return NUM_PRICE_BUCKETS - 1

    def _get_duration_bucket(self, time_str: str) -> int:
        """Convert duration string to bucket index (0-6)."""
        import re
        if 'ngày' in time_str:
            match = re.search(r'(\d+)\s*ngày', time_str)
            if match:
                days = int(match.group(1))
                return min(days - 1, NUM_DURATION_BUCKETS - 1)
        return 0

    def _get_season(self) -> int:
        """Get current season (0=spring, 1=summer, 2=fall, 3=winter)."""
        month = datetime.now().month
        if month in [3, 4, 5]:
            return 0  # Spring
        elif month in [6, 7, 8]:
            return 1  # Summer
        elif month in [9, 10, 11]:
            return 2  # Fall
        else:
            return 3  # Winter

    def _create_model(self) -> None:
        """Create DeepFM model."""
        self.model = DeepFMModel(
            num_users=len(self.user_encoder) + 100,  # Buffer for new users
            num_tours=len(self.tour_encoder) + 50,   # Buffer for new tours
            num_destinations=len(self.dest_encoder) + 10,
            embedding_dim=self.embedding_dim,
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

    async def train(self, db, epochs: int = 15) -> Dict:
        """Train model on booking/review data."""
        logger.info("Training DeepFM model...")

        # Prepare training data
        X, y = await self._prepare_training_data(db)

        if len(y) < 10:
            logger.warning("Not enough training data, skipping training")
            return {"status": "skipped", "reason": "insufficient_data"}

        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )

        self.is_trained = True
        self.last_update = datetime.now()

        # Save model
        self._save_model()

        metrics = {
            "status": "success",
            "epochs": len(history.history['loss']),
            "final_loss": float(history.history['loss'][-1]),
            "final_auc": float(history.history.get('auc', [0])[-1]),
            "samples": len(y)
        }

        logger.info(f"Training completed: {metrics}")
        return metrics

    async def _prepare_training_data(self, db) -> Tuple[Dict, np.ndarray]:
        """Prepare training data from bookings and reviews."""
        # Get bookings (positive samples)
        bookings = await db.tbl_booking.find(
            {'bookingStatus': {'$ne': 'x'}}
        ).to_list(10000)

        # Get reviews
        reviews = await db.tbl_reviews.find({}).to_list(10000)

        # Get tour info
        tours = await db.tbl_tours.find({}).to_list(1000)
        tour_info = {str(t['_id']): t for t in tours}

        # Get departures for tour_id mapping
        departures = await db.tbl_tour_departures.find({}).to_list(5000)
        dep_to_tour = {str(d['_id']): str(d.get('tourId', '')) for d in departures}

        samples = []
        current_season = self._get_season()

        # Process bookings (label = 1 for confirmed, 0.7 for pending)
        for booking in bookings:
            user_id = str(booking.get('userId', ''))
            dep_id = str(booking.get('tourDepartureId', ''))
            tour_id = dep_to_tour.get(dep_id, str(booking.get('tourId', '')))

            if user_id not in self.user_encoder or tour_id not in self.tour_encoder:
                continue

            tour = tour_info.get(tour_id, {})
            label = 1.0 if booking.get('bookingStatus') == 'c' else 0.7

            samples.append(self._create_sample(user_id, tour_id, tour, label, current_season))

        # Process reviews (label based on rating)
        for review in reviews:
            user_id = str(review.get('userId', ''))
            tour_id = str(review.get('tourId', ''))

            if user_id not in self.user_encoder or tour_id not in self.tour_encoder:
                continue

            tour = tour_info.get(tour_id, {})
            rating = review.get('rating', 3)
            label = rating / 5.0  # Normalize to 0-1

            samples.append(self._create_sample(user_id, tour_id, tour, label, current_season))

        # Add negative samples (tours user didn't interact with)
        all_tour_ids = list(self.tour_encoder.keys())
        user_positive_tours = {}

        for s in samples:
            uid = s['user_id']
            tid = s['tour_id']
            if uid not in user_positive_tours:
                user_positive_tours[uid] = set()
            user_positive_tours[uid].add(tid)

        for user_id, user_idx in list(self.user_encoder.items())[:100]:
            positive_tours = user_positive_tours.get(user_idx, set())
            neg_tours = [t for t in all_tour_ids if self.tour_encoder[t] not in positive_tours]

            if not neg_tours:
                continue

            neg_sample_size = min(5, len(neg_tours))
            sampled_neg = np.random.choice(neg_tours, neg_sample_size, replace=False)

            for tour_id in sampled_neg:
                tour = tour_info.get(tour_id, {})
                samples.append(self._create_sample(user_id, tour_id, tour, 0.0, current_season))

        # Shuffle and convert to arrays
        np.random.shuffle(samples)

        X = {
            'user_id': np.array([s['user_id'] for s in samples]),
            'tour_id': np.array([s['tour_id'] for s in samples]),
            'destination_id': np.array([s['destination_id'] for s in samples]),
            'price_bucket': np.array([s['price_bucket'] for s in samples]),
            'duration_bucket': np.array([s['duration_bucket'] for s in samples]),
            'has_images': np.array([s['has_images'] for s in samples]),
            'has_itinerary': np.array([s['has_itinerary'] for s in samples]),
            'season': np.array([s['season'] for s in samples]),
        }
        y = np.array([s['label'] for s in samples])

        logger.info(f"Prepared {len(samples)} training samples")
        return X, y

    def _create_sample(self, user_id: str, tour_id: str, tour: Dict,
                       label: float, season: int) -> Dict:
        """Create a single training sample with all 8 fields."""
        return {
            'user_id': self.user_encoder.get(user_id, 0),
            'tour_id': self.tour_encoder.get(tour_id, 0),
            'destination_id': self.dest_encoder.get(tour.get('destination', ''), 0),
            'price_bucket': self._get_price_bucket(tour.get('priceAdult', 0)),
            'duration_bucket': self._get_duration_bucket(tour.get('time', '')),
            'has_images': 1 if tour.get('images') else 0,
            'has_itinerary': 1 if tour.get('itinerary') else 0,
            'season': season,
            'label': label
        }

    async def predict(
        self,
        user_id: str,
        tour_ids: List[str],
        db
    ) -> List[Tuple[str, float]]:
        """Predict scores for user-tour pairs."""
        if not self.is_trained or self.model is None:
            return [(tid, 0.5) for tid in tour_ids]

        # Get tour info
        from bson import ObjectId
        tours = await db.tbl_tours.find({
            '_id': {'$in': [ObjectId(tid) for tid in tour_ids]}
        }).to_list(100)
        tour_info = {str(t['_id']): t for t in tours}

        # Prepare batch input
        samples = []
        valid_tour_ids = []
        current_season = self._get_season()

        for tour_id in tour_ids:
            tour = tour_info.get(tour_id, {})
            if not tour:
                continue

            sample = self._create_sample(user_id, tour_id, tour, 0.0, current_season)
            del sample['label']
            samples.append(sample)
            valid_tour_ids.append(tour_id)

        if not samples:
            return []

        X = {
            'user_id': np.array([s['user_id'] for s in samples]),
            'tour_id': np.array([s['tour_id'] for s in samples]),
            'destination_id': np.array([s['destination_id'] for s in samples]),
            'price_bucket': np.array([s['price_bucket'] for s in samples]),
            'duration_bucket': np.array([s['duration_bucket'] for s in samples]),
            'has_images': np.array([s['has_images'] for s in samples]),
            'has_itinerary': np.array([s['has_itinerary'] for s in samples]),
            'season': np.array([s['season'] for s in samples]),
        }

        # Predict
        scores = self.model.predict(X, verbose=0).flatten()

        return list(zip(valid_tour_ids, scores.tolist()))

    async def online_update(self, db, interaction: Dict) -> None:
        """
        Update model with new interaction (online learning).

        interaction: {
            'user_id': str,
            'tour_id': str,
            'label': float (0-1),
            'type': 'booking' | 'review' | 'click'
        }
        """
        if not self.is_trained or self.model is None:
            return

        user_id = interaction['user_id']
        tour_id = interaction['tour_id']
        label = interaction['label']

        # Add new user/tour to encoder if needed
        if user_id not in self.user_encoder:
            self.user_encoder[user_id] = len(self.user_encoder) + 1
        if tour_id not in self.tour_encoder:
            self.tour_encoder[tour_id] = len(self.tour_encoder) + 1

        # Get tour info
        from bson import ObjectId
        tour = await db.tbl_tours.find_one({'_id': ObjectId(tour_id)})

        if not tour:
            return

        # Prepare single sample
        sample = self._create_sample(user_id, tour_id, tour, label, self._get_season())

        X = {
            'user_id': np.array([sample['user_id']]),
            'tour_id': np.array([sample['tour_id']]),
            'destination_id': np.array([sample['destination_id']]),
            'price_bucket': np.array([sample['price_bucket']]),
            'duration_bucket': np.array([sample['duration_bucket']]),
            'has_images': np.array([sample['has_images']]),
            'has_itinerary': np.array([sample['has_itinerary']]),
            'season': np.array([sample['season']]),
        }
        y = np.array([label])

        # Single step update
        self.model.fit(X, y, epochs=1, verbose=0)

        logger.debug(f"Online update: user={user_id}, tour={tour_id}, label={label}")

    def _save_model(self) -> None:
        """Save model and encoders."""
        os.makedirs(self.model_path, exist_ok=True)

        # Save model weights
        self.model.save_weights(f"{self.model_path}/weights.h5")

        # Save encoders and config
        config = {
            'user_encoder': self.user_encoder,
            'tour_encoder': self.tour_encoder,
            'dest_encoder': self.dest_encoder,
            'price_boundaries': self.price_boundaries,
            'is_trained': self.is_trained,
            'last_update': self.last_update,
            'model_config': {
                'num_users': self.model.num_users,
                'num_tours': self.model.num_tours,
                'num_destinations': self.model.num_destinations,
                'embedding_dim': self.model.embedding_dim,
            }
        }

        with open(f"{self.model_path}/config.pkl", 'wb') as f:
            pickle.dump(config, f)

        logger.info(f"Model saved to {self.model_path}")

    def _load_model(self) -> bool:
        """Load model and encoders."""
        config_path = f"{self.model_path}/config.pkl"
        weights_path = f"{self.model_path}/weights.h5"

        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            return False

        try:
            with open(config_path, 'rb') as f:
                config = pickle.load(f)

            self.user_encoder = config['user_encoder']
            self.tour_encoder = config['tour_encoder']
            self.dest_encoder = config['dest_encoder']
            self.price_boundaries = config.get('price_boundaries', [])
            self.is_trained = config['is_trained']
            self.last_update = config['last_update']

            # Recreate model with same config
            model_config = config['model_config']
            self.model = DeepFMModel(
                num_users=model_config['num_users'],
                num_tours=model_config['num_tours'],
                num_destinations=model_config['num_destinations'],
                embedding_dim=model_config['embedding_dim'],
            )

            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc')]
            )

            # Build model by calling with dummy input
            dummy_input = {
                'user_id': np.array([0]),
                'tour_id': np.array([0]),
                'destination_id': np.array([0]),
                'price_bucket': np.array([0]),
                'duration_bucket': np.array([0]),
                'has_images': np.array([0]),
                'has_itinerary': np.array([0]),
                'season': np.array([0]),
            }
            self.model(dummy_input)

            # Load weights
            self.model.load_weights(weights_path)

            logger.info(f"Model loaded from {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
