"""
User Interaction Tracking Service

Lưu trữ và phân tích hành vi người dùng để:
1. Online learning cho DeepFM
2. Analytics và reporting
3. A/B testing
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from bson import ObjectId

logger = logging.getLogger(__name__)


class InteractionService:
    """
    Service để tracking và lưu trữ user interactions.

    Collection: tbl_user_interactions
    Schema:
    {
        _id: ObjectId,
        userId: ObjectId,
        tourId: ObjectId,
        type: "view" | "click" | "bookmark" | "share" | "booking" | "review",
        value: float,          # rating cho review, 1.0 cho các loại khác
        source: str,           # "homepage" | "similar" | "post_booking" | "search" | "direct"
        model: str,            # "deepfm" | "hybrid" | "popularity"
        position: int,         # Vị trí trong danh sách recommendation (0-indexed)
        sessionId: str,        # Session ID để group interactions
        deviceType: str,       # "mobile" | "desktop" | "tablet"
        duration: int,         # Thời gian xem (giây) - cho type="view"
        createdAt: datetime,
        metadata: {}           # Additional data
    }
    """

    def __init__(self, db):
        self.db = db
        self.collection = db.tbl_user_interactions

    async def ensure_indexes(self):
        """Create indexes for efficient queries."""
        await self.collection.create_index("userId")
        await self.collection.create_index("tourId")
        await self.collection.create_index("type")
        await self.collection.create_index("createdAt")
        await self.collection.create_index([("userId", 1), ("tourId", 1), ("type", 1)])
        await self.collection.create_index([("source", 1), ("model", 1)])
        await self.collection.create_index("sessionId")
        logger.info("Interaction indexes created")

    async def track(
        self,
        user_id: str,
        tour_id: str,
        interaction_type: str,
        value: float = 1.0,
        source: str = "direct",
        model: str = None,
        position: int = None,
        session_id: str = None,
        device_type: str = "desktop",
        duration: int = None,
        metadata: Dict = None
    ) -> str:
        """
        Track một user interaction.

        Args:
            user_id: ID người dùng
            tour_id: ID tour
            interaction_type: Loại tương tác (view, click, bookmark, share, booking, review)
            value: Giá trị (rating cho review, 1.0 mặc định)
            source: Nguồn recommendation (homepage, similar, post_booking, search, direct)
            model: Model đã dùng (deepfm, hybrid, popularity)
            position: Vị trí trong danh sách (0-indexed)
            session_id: Session ID
            device_type: Loại thiết bị
            duration: Thời gian xem (giây)
            metadata: Data bổ sung

        Returns:
            ID của interaction đã tạo
        """
        doc = {
            "userId": ObjectId(user_id) if user_id else None,
            "tourId": ObjectId(tour_id),
            "type": interaction_type,
            "value": value,
            "source": source,
            "model": model,
            "position": position,
            "sessionId": session_id,
            "deviceType": device_type,
            "duration": duration,
            "createdAt": datetime.utcnow(),
            "metadata": metadata or {}
        }

        result = await self.collection.insert_one(doc)
        logger.debug(f"Tracked: user={user_id}, tour={tour_id}, type={interaction_type}")

        return str(result.inserted_id)

    async def get_user_history(
        self,
        user_id: str,
        interaction_types: List[str] = None,
        limit: int = 100,
        days: int = 30
    ) -> List[Dict]:
        """Lấy lịch sử tương tác của user."""
        query = {
            "userId": ObjectId(user_id),
            "createdAt": {"$gte": datetime.utcnow() - timedelta(days=days)}
        }

        if interaction_types:
            query["type"] = {"$in": interaction_types}

        cursor = self.collection.find(query).sort("createdAt", -1).limit(limit)
        return await cursor.to_list(limit)

    async def get_tour_interactions(
        self,
        tour_id: str,
        days: int = 30
    ) -> Dict:
        """Thống kê tương tác của một tour."""
        pipeline = [
            {
                "$match": {
                    "tourId": ObjectId(tour_id),
                    "createdAt": {"$gte": datetime.utcnow() - timedelta(days=days)}
                }
            },
            {
                "$group": {
                    "_id": "$type",
                    "count": {"$sum": 1},
                    "unique_users": {"$addToSet": "$userId"}
                }
            }
        ]

        results = await self.collection.aggregate(pipeline).to_list(10)

        stats = {}
        for r in results:
            stats[r["_id"]] = {
                "count": r["count"],
                "unique_users": len(r["unique_users"])
            }

        return stats

    async def get_recommendation_metrics(
        self,
        model: str = None,
        source: str = None,
        days: int = 7
    ) -> Dict:
        """
        Tính các metrics cho recommendation system.

        Returns:
            - impressions: Số lần hiển thị
            - clicks: Số click
            - ctr: Click-through rate
            - bookings: Số booking từ recommendations
            - conversion_rate: Tỷ lệ chuyển đổi
        """
        match_stage = {
            "createdAt": {"$gte": datetime.utcnow() - timedelta(days=days)}
        }

        if model:
            match_stage["model"] = model
        if source:
            match_stage["source"] = source

        # Exclude direct access
        if not source:
            match_stage["source"] = {"$ne": "direct"}

        pipeline = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": {
                        "model": "$model",
                        "source": "$source"
                    },
                    "impressions": {
                        "$sum": {"$cond": [{"$eq": ["$type", "view"]}, 1, 0]}
                    },
                    "clicks": {
                        "$sum": {"$cond": [{"$eq": ["$type", "click"]}, 1, 0]}
                    },
                    "bookings": {
                        "$sum": {"$cond": [{"$eq": ["$type", "booking"]}, 1, 0]}
                    },
                    "unique_users": {"$addToSet": "$userId"},
                    "avg_position": {
                        "$avg": {
                            "$cond": [
                                {"$and": [
                                    {"$eq": ["$type", "click"]},
                                    {"$ne": ["$position", None]}
                                ]},
                                "$position",
                                None
                            ]
                        }
                    }
                }
            }
        ]

        results = await self.collection.aggregate(pipeline).to_list(20)

        metrics = {}
        for r in results:
            key = f"{r['_id']['model']}_{r['_id']['source']}"
            impressions = r["impressions"] or 1
            clicks = r["clicks"]
            bookings = r["bookings"]

            metrics[key] = {
                "model": r["_id"]["model"],
                "source": r["_id"]["source"],
                "impressions": impressions,
                "clicks": clicks,
                "bookings": bookings,
                "ctr": round(clicks / impressions * 100, 2) if impressions > 0 else 0,
                "conversion_rate": round(bookings / clicks * 100, 2) if clicks > 0 else 0,
                "unique_users": len(r["unique_users"]),
                "avg_click_position": round(r["avg_position"], 2) if r["avg_position"] else None
            }

        return metrics

    async def get_ab_test_results(
        self,
        days: int = 7
    ) -> Dict:
        """So sánh hiệu quả giữa các models."""
        metrics = await self.get_recommendation_metrics(days=days)

        # Group by model
        model_stats = {}
        for key, data in metrics.items():
            model = data["model"]
            if model not in model_stats:
                model_stats[model] = {
                    "impressions": 0,
                    "clicks": 0,
                    "bookings": 0,
                    "unique_users": set()
                }

            model_stats[model]["impressions"] += data["impressions"]
            model_stats[model]["clicks"] += data["clicks"]
            model_stats[model]["bookings"] += data["bookings"]

        # Calculate aggregate metrics
        results = {}
        for model, stats in model_stats.items():
            impressions = stats["impressions"] or 1
            clicks = stats["clicks"]
            bookings = stats["bookings"]

            results[model] = {
                "impressions": impressions,
                "clicks": clicks,
                "bookings": bookings,
                "ctr": round(clicks / impressions * 100, 2),
                "conversion_rate": round(bookings / clicks * 100, 2) if clicks > 0 else 0
            }

        return results

    async def get_daily_stats(self, days: int = 7) -> List[Dict]:
        """Thống kê theo ngày."""
        pipeline = [
            {
                "$match": {
                    "createdAt": {"$gte": datetime.utcnow() - timedelta(days=days)}
                }
            },
            {
                "$group": {
                    "_id": {
                        "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$createdAt"}},
                        "type": "$type"
                    },
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id.date": 1}}
        ]

        results = await self.collection.aggregate(pipeline).to_list(100)

        # Reshape data
        daily_data = {}
        for r in results:
            date = r["_id"]["date"]
            interaction_type = r["_id"]["type"]

            if date not in daily_data:
                daily_data[date] = {"date": date}

            daily_data[date][interaction_type] = r["count"]

        return list(daily_data.values())
