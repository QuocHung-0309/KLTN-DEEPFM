"""
DeepFM Model using LibRecommender library.

Thay thế custom TF/Keras implementation bằng LibRecommender để:
- Dùng FM + Deep đã được tối ưu sẵn
- Tự động negative sampling
- Đánh giá metrics chuẩn (roc_auc, ndcg, precision, recall)

Features:
- destination (sparse, item)
- price_bucket (sparse, item) — 10 buckets theo quantile
- duration_bucket (sparse, item) — 0..6 (số ngày tour)

Label sources (max label per user-tour pair):
- tbl_reviews      → rating 1-5
- tbl_booking      → completed=5, confirmed=4, pending=3
- tbl_user_interactions → bookmark=4, share=3.5, click=2, view=1
"""

import io
import logging
import os
import pickle
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from bson import ObjectId

logger = logging.getLogger(__name__)

NUM_PRICE_BUCKETS = 10
NUM_DURATION_BUCKETS = 7

BOOKING_LABELS  = {"completed": 5.0, "confirmed": 4.0, "pending": 3.0}
INTERACT_LABELS = {"bookmark": 4.0, "share": 3.5, "click": 2.0, "view": 1.0}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_days(time_str: str) -> int:
    if not time_str:
        return 1
    s = time_str.lower()
    m = re.search(r"(\d+)\s*ng[aà]y", s)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*n", s)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", time_str)
    if m:
        return int(m.group(1))
    return 1


def _day_bucket(days: int) -> int:
    return min(max(days - 1, 0), 6)


# ── Main class ────────────────────────────────────────────────────────────────

class DeepFMRecommender:
    """
    DeepFM recommender built on LibRecommender.

    Public interface giữ nguyên so với custom model cũ để
    DeepFMRecommenderService không cần sửa.
    """

    def __init__(
        self,
        model_path: str = "models/deepfm_libreco",
        n_epochs: int = 20,
        embed_size: int = 16,
        batch_size: int = 256,
        lr: float = 1e-3,
    ):
        self.model_path = model_path
        self.n_epochs = n_epochs
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.lr = lr

        self.model = None
        self.data_info = None

        # Encoder mappings (MongoDB string ID → libreco int)
        self.user_encoder: Dict[str, int] = {}
        self.tour_encoder: Dict[str, int] = {}
        self.tour_decoder: Dict[int, str] = {}
        self.dest_encoder: Dict[str, int] = {}   # kept for API compat
        self.price_boundaries: List[float] = []

        self.is_trained = False
        self.last_update: Optional[datetime] = None
        self.training_history: List[Dict] = []

        # Buffer cho online-style updates (retrain khi đủ ngưỡng)
        self._pending: List[Dict] = []
        self._retrain_threshold = 100

    # ── Initialize / Load ────────────────────────────────────────────────────

    async def initialize(self, db) -> None:
        if self._load_model():
            logger.info("LibRecommender DeepFM loaded from disk")
            self.is_trained = True
        else:
            logger.info("No saved model — training from scratch")
            await self.train(db)

    # ── Data preparation ──────────────────────────────────────────────────────

    async def _fetch_and_build_df(self, db) -> pd.DataFrame:
        """Load data từ MongoDB và trả về DataFrame theo format LibRecommender."""

        tours_raw = await db.tbl_tours.find(
            {}, {"_id": 1, "destination": 1, "priceAdult": 1, "time": 1}
        ).to_list(1000)

        deps_raw = await db.tbl_tour_departures.find(
            {}, {"_id": 1, "tourId": 1}
        ).to_list(5000)

        reviews_raw = await db.tbl_reviews.find(
            {}, {"_id": 1, "userId": 1, "tourId": 1, "rating": 1}
        ).to_list(10000)

        bookings_raw = await db.tbl_booking.find(
            {"bookingStatus": {"$in": ["completed", "confirmed", "pending"]}},
            {"_id": 1, "userId": 1, "tourDepartureId": 1, "bookingStatus": 1}
        ).to_list(10000)

        interactions_raw = await db.tbl_user_interactions.find(
            {"type": {"$in": ["view", "click", "bookmark", "share"]}},
            {"_id": 1, "userId": 1, "tourId": 1, "type": 1}
        ).to_list(50000)

        if not tours_raw:
            return pd.DataFrame()

        dep_to_tour = {str(d["_id"]): str(d["tourId"]) for d in deps_raw}
        tour_info   = {str(t["_id"]): t for t in tours_raw}

        # Price boundaries
        prices = [t.get("priceAdult", 0) for t in tours_raw if t.get("priceAdult", 0) > 0]
        self.price_boundaries = (
            [np.percentile(prices, p) for p in range(10, 100, 10)]
            if prices else [1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 8e6, 10e6, 15e6]
        )

        rows = []
        for r in reviews_raw:
            uid = str(r.get("userId", ""))
            tid = str(r.get("tourId", ""))
            if uid and tid and tid in tour_info:
                rows.append({"user_str": uid, "tour_id_str": tid,
                             "label": float(r.get("rating", 3))})

        for b in bookings_raw:
            uid = str(b.get("userId", ""))
            dep_id = str(b.get("tourDepartureId", ""))
            tid = dep_to_tour.get(dep_id, "")
            label = BOOKING_LABELS.get(b.get("bookingStatus", ""), 3.0)
            if uid and tid and tid in tour_info:
                rows.append({"user_str": uid, "tour_id_str": tid, "label": label})

        for i in interactions_raw:
            uid = str(i.get("userId", ""))
            tid = str(i.get("tourId", ""))
            label = INTERACT_LABELS.get(i.get("type", "view"), 1.0)
            if uid and tid and tid in tour_info:
                rows.append({"user_str": uid, "tour_id_str": tid, "label": label})

        if not rows:
            return pd.DataFrame()

        agg = (
            pd.DataFrame(rows)
            .groupby(["user_str", "tour_id_str"])["label"]
            .max()
            .reset_index()
        )

        # Encode IDs
        unique_users = agg["user_str"].unique()
        unique_tours = agg["tour_id_str"].unique()
        self.user_encoder = {u: i for i, u in enumerate(unique_users)}
        self.tour_encoder = {t: i for i, t in enumerate(unique_tours)}
        self.tour_decoder = {v: k for k, v in self.tour_encoder.items()}
        all_dests = set(t.get("destination", "unknown") or "unknown" for t in tours_raw)
        self.dest_encoder = {d: i for i, d in enumerate(all_dests)}

        agg["user"] = agg["user_str"].map(self.user_encoder)
        agg["item"] = agg["tour_id_str"].map(self.tour_encoder)

        # Tour features
        feat_rows = []
        for t in tours_raw:
            tid = str(t["_id"])
            feat_rows.append({
                "tour_id_str": tid,
                "destination":     t.get("destination", "unknown") or "unknown",
                "price_bucket":    self._price_bucket(t.get("priceAdult", 0)),
                "duration_bucket": _day_bucket(_parse_days(t.get("time", ""))),
            })

        feat_df = pd.DataFrame(feat_rows)
        feat_df["item"] = feat_df["tour_id_str"].map(self.tour_encoder)
        feat_df = feat_df.dropna(subset=["item"])
        feat_df["item"] = feat_df["item"].astype(int)

        data = agg.merge(
            feat_df[["item", "destination", "price_bucket", "duration_bucket"]],
            on="item", how="left"
        )
        data["destination"]     = data["destination"].fillna("unknown")
        data["price_bucket"]    = data["price_bucket"].fillna(0).astype(int)
        data["duration_bucket"] = data["duration_bucket"].fillna(0).astype(int)
        data = data[["user", "item", "label",
                     "destination", "price_bucket", "duration_bucket"]].dropna()

        logger.info(
            f"Training data: {len(data)} samples | "
            f"{data['user'].nunique()} users | {data['item'].nunique()} tours"
        )
        return data

    @staticmethod
    def _parse_training_output(output: str) -> List[Dict]:
        """Parse LibRecommender stdout into per-epoch metric dicts."""
        history: List[Dict] = []
        current: Dict = {}
        for line in output.split("\n"):
            m = re.match(r"(?i)epoch\s+(\d+)", line.strip())
            if m:
                if current:
                    history.append(current.copy())
                current = {"epoch": int(m.group(1))}
            if not current:
                continue
            for key, pat in [
                ("loss",      r"(?:train_)?loss[\s:=]+([0-9.]+)"),
                ("roc_auc",   r"roc_auc[\w@]*[\s:=]+([0-9.]+)"),
                ("precision", r"precision[\w@]*[\s:=]+([0-9.]+)"),
                ("recall",    r"recall[\w@]*[\s:=]+([0-9.]+)"),
                ("ndcg",      r"ndcg[\w@]*[\s:=]+([0-9.]+)"),
            ]:
                if key not in current:
                    vm = re.search(pat, line, re.I)
                    if vm:
                        current[key] = float(vm.group(1))
        if current:
            history.append(current)
        return history

    def _price_bucket(self, price: float) -> int:
        for i, b in enumerate(self.price_boundaries):
            if price <= b:
                return i
        return NUM_PRICE_BUCKETS - 1

    # ── Train ─────────────────────────────────────────────────────────────────

    async def train(self, db, epochs: int = None) -> Dict:
        from libreco.algorithms import DeepFM as LibrecoDeepFM
        from libreco.data import DatasetFeat, random_split as lib_split

        n_ep = epochs or self.n_epochs
        logger.info(f"Training LibRecommender DeepFM ({n_ep} epochs)...")

        data = await self._fetch_and_build_df(db)
        if len(data) < 10:
            logger.warning("Insufficient training data")
            return {"status": "skipped", "reason": "insufficient_data"}

        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split
        if len(data) >= 30:
            train_df, eval_df = lib_split(data, multi_ratios=[0.9, 0.1], seed=42)
        else:
            train_df, eval_df = data, None

        tf.compat.v1.reset_default_graph()

        train_df, data_info = DatasetFeat.build_trainset(
            train_df,
            user_col=[],
            item_col=["destination", "price_bucket", "duration_bucket"],
            sparse_col=["destination", "price_bucket", "duration_bucket"],
            dense_col=[],
        )
        self.data_info = data_info

        if eval_df is not None:
            eval_df = DatasetFeat.build_evalset(eval_df)

        self.model = LibrecoDeepFM(
            task="ranking",
            data_info=data_info,
            embed_size=self.embed_size,
            n_epochs=n_ep,
            loss_type="cross_entropy",
            lr=self.lr,
            lr_decay=False,
            reg=None,
            batch_size=self.batch_size,
            use_bn=True,
            dropout_rate=0.2,
            hidden_units=(128, 64, 32),
            multi_sparse_combiner="sqrtn",
            sampler="random",
            num_neg=1,
            seed=42,
        )

        fit_kw = {"neg_sampling": True, "verbose": 1, "shuffle": True}
        if eval_df is not None:
            fit_kw["eval_data"] = eval_df
            fit_kw["metrics"] = ["loss", "roc_auc", "precision", "recall", "ndcg"]

        buf = io.StringIO()
        _old = sys.stdout
        sys.stdout = buf
        try:
            self.model.fit(train_df, **fit_kw)
        finally:
            sys.stdout = _old

        raw_output = buf.getvalue()
        print(raw_output, end="")
        self.training_history = self._parse_training_output(raw_output)

        self.is_trained = True
        self.last_update = datetime.now()
        self._save_model()

        result = {
            "status": "success",
            "samples": int(len(data)),
            "n_users": int(data["user"].nunique()),
            "n_tours": int(data["item"].nunique()),
            "epochs": n_ep,
            "last_update": str(self.last_update),
        }
        logger.info(f"Training done: {result}")
        return result

    # ── Predict ───────────────────────────────────────────────────────────────

    async def predict(
        self,
        user_id: str,
        tour_ids: List[str],
        db,
    ) -> List[Tuple[str, float]]:
        """Trả về list (tour_id_str, score) cho các tour được yêu cầu."""
        if not self.is_trained or self.model is None:
            return [(tid, 0.5) for tid in tour_ids]

        encoded_user = self.user_encoder.get(user_id)

        # Cold-start user: dùng recommend_user với cold_start="average"
        if encoded_user is None:
            return self._cold_start_scores(tour_ids)

        valid = [(tid, self.tour_encoder[tid])
                 for tid in tour_ids if tid in self.tour_encoder]

        if not valid:
            return [(tid, 0.5) for tid in tour_ids]

        str_ids  = [v[0] for v in valid]
        item_ids = [v[1] for v in valid]

        try:
            scores = self.model.predict(
                user=[encoded_user] * len(item_ids),
                item=item_ids,
            )
            score_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
            return list(zip(str_ids, score_list))
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return [(tid, 0.5) for tid in tour_ids]

    def _cold_start_scores(self, tour_ids: List[str]) -> List[Tuple[str, float]]:
        """Score cho cold-start user: dùng popularity proxy (trả về 0.5 uniform)."""
        return [(tid, 0.5) for tid in tour_ids]

    async def recommend(self, user_id: str, n_rec: int = 10) -> List[str]:
        """
        Đề xuất top-N tour (trả về list MongoDB tour ID string).
        Dùng LibRecommender recommend_user() — tự loại tour đã tương tác.
        """
        if not self.is_trained or self.model is None:
            return []

        encoded_user = self.user_encoder.get(user_id)
        if encoded_user is None:
            return []

        try:
            recs = self.model.recommend_user(user=encoded_user, n_rec=n_rec)
            encoded_items = recs.get(encoded_user, [])
            return [self.tour_decoder[eid]
                    for eid in encoded_items if eid in self.tour_decoder]
        except Exception as e:
            logger.error(f"Recommend error: {e}")
            return []

    # ── Online update ─────────────────────────────────────────────────────────

    async def online_update(self, db, interaction: Dict) -> None:
        """
        Tích lũy interaction mới, retrain khi đạt ngưỡng.
        LibRecommender không hỗ trợ incremental update nên dùng batch retrain.
        """
        self._pending.append(interaction)
        logger.debug(f"Pending interactions: {len(self._pending)}/{self._retrain_threshold}")

        if len(self._pending) >= self._retrain_threshold:
            logger.info("Threshold reached — retraining DeepFM...")
            await self.train(db)
            self._pending.clear()

    # ── Save / Load ───────────────────────────────────────────────────────────

    def _save_model(self) -> None:
        os.makedirs(self.model_path, exist_ok=True)

        self.data_info.save(self.model_path, model_name="deepfm")
        self.model.save(self.model_path, model_name="deepfm")

        meta = {
            "user_encoder":    self.user_encoder,
            "tour_encoder":    self.tour_encoder,
            "tour_decoder":    self.tour_decoder,
            "dest_encoder":    self.dest_encoder,
            "price_boundaries": self.price_boundaries,
            "is_trained":      self.is_trained,
            "last_update":     self.last_update,
            "n_epochs":        self.n_epochs,
            "embed_size":      self.embed_size,
            "training_history": self.training_history,
        }
        with open(f"{self.model_path}/meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        logger.info(f"Model saved → {self.model_path}")

    def _load_model(self) -> bool:
        meta_path = f"{self.model_path}/meta.pkl"
        if not os.path.exists(meta_path):
            return False

        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)

            self.user_encoder    = meta["user_encoder"]
            self.tour_encoder    = meta["tour_encoder"]
            self.tour_decoder    = meta["tour_decoder"]
            self.dest_encoder    = meta.get("dest_encoder", {})
            self.price_boundaries  = meta["price_boundaries"]
            self.last_update       = meta["last_update"]
            self.training_history  = meta.get("training_history", [])

            from libreco.algorithms import DeepFM as LibrecoDeepFM
            from libreco.data import DataInfo

            tf.compat.v1.reset_default_graph()
            self.data_info = DataInfo.load(self.model_path, model_name="deepfm")
            self.model = LibrecoDeepFM.load(
                path=self.model_path,
                model_name="deepfm",
                data_info=self.data_info,
            )
            logger.info("Model loaded from disk")
            return True

        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False
