from fastapi import APIRouter, Request, Query, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, List
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class TrackingRequest(BaseModel):
    """Request body for tracking interactions."""
    userId: Optional[str] = None
    tourId: str
    type: str  # view, click, bookmark, share, booking, review
    value: float = 1.0
    source: str = "direct"  # homepage, similar, post_booking, search, direct
    model: Optional[str] = None  # deepfm, hybrid, popularity
    position: Optional[int] = None  # Position in recommendation list
    sessionId: Optional[str] = None
    deviceType: str = "desktop"  # mobile, desktop, tablet
    duration: Optional[int] = None  # View duration in seconds


def format_tour_response(tour: dict) -> dict:
    """Format tour document for API response matching frontend expectations."""
    return {
        "_id": str(tour.get("_id", "")),
        "title": tour.get("title", ""),
        "destination": tour.get("destination", ""),
        "destinationSlug": tour.get("destinationSlug", ""),
        "priceAdult": tour.get("priceAdult", 0),
        "priceChild": tour.get("priceChild", 0),
        "salePrice": tour.get("salePrice"),
        "discountPercent": tour.get("discountPercent"),
        "time": tour.get("time", ""),
        "description": tour.get("description", ""),
        "images": tour.get("images", []),
        "quantity": tour.get("quantity"),
        "startDate": tour.get("startDate"),
        "upcomingDepartures": tour.get("upcomingDepartures", []),
    }


def get_recommender(request: Request, prefer_deepfm: bool = True):
    """Get the best available recommender (DeepFM or Hybrid)."""
    if prefer_deepfm:
        deepfm = getattr(request.app.state, "deepfm_recommender", None)
        if deepfm and deepfm.is_initialized:
            return deepfm, "deepfm"

    hybrid = getattr(request.app.state, "recommender", None)
    if hybrid:
        return hybrid, "hybrid"

    return None, None


@router.get("/homepage")
async def get_homepage_recommendations(
    request: Request,
    userId: Optional[str] = Query(None, description="User ID for personalization"),
    limit: int = Query(6, ge=1, le=20, description="Number of recommendations"),
    model: Optional[str] = Query(None, description="Force specific model: 'deepfm' or 'hybrid'")
):
    """
    Personalized homepage recommendations using DeepFM or Hybrid model.

    - With userId: Personalized recommendations based on user history
    - Without userId: Popular tours fallback

    Query params:
    - model: Force a specific model ('deepfm' or 'hybrid')
    """
    try:
        tour_service = getattr(request.app.state, "tour_service", None)

        if not tour_service:
            logger.error("Tour service not initialized")
            return {"data": [], "model": None}

        # Select recommender based on query param or default preference
        if model == "deepfm":
            recommender = getattr(request.app.state, "deepfm_recommender", None)
            model_used = "deepfm" if recommender else None
        elif model == "hybrid":
            recommender = getattr(request.app.state, "recommender", None)
            model_used = "hybrid" if recommender else None
        else:
            recommender, model_used = get_recommender(request, prefer_deepfm=True)

        # If no user ID or no recommender, return popular tours
        if not userId or not recommender:
            logger.info(f"Returning popular tours (userId={userId}, model={model_used})")
            popular = await tour_service.get_popular_tours(limit)
            return {"data": [format_tour_response(t) for t in popular], "model": "popularity"}

        # Get personalized recommendations
        logger.info(f"Getting personalized recommendations for user {userId} using {model_used}")

        if model_used == "deepfm":
            tours = await recommender.get_homepage_recommendations(userId, limit)
        else:
            tours = await recommender.get_homepage_recommendations(userId, limit)

        # Fallback to popular if no recommendations
        if not tours:
            logger.info(f"No personalized recommendations, falling back to popular")
            tours = await tour_service.get_popular_tours(limit)
            model_used = "popularity"

        return {"data": [format_tour_response(t) for t in tours], "model": model_used}

    except Exception as e:
        logger.error(f"Homepage recommendation error: {e}", exc_info=True)

        # Graceful degradation: try to return popular tours
        try:
            tour_service = getattr(request.app.state, "tour_service", None)
            if tour_service:
                popular = await tour_service.get_popular_tours(limit)
                return {"data": [format_tour_response(t) for t in popular], "model": "popularity"}
        except Exception:
            pass

        return {"data": [], "model": None}


@router.get("/similar")
async def get_similar_tours(
    request: Request,
    tourId: str = Query(..., description="Tour ID to find similar tours for"),
    limit: int = Query(4, ge=1, le=10, description="Number of similar tours"),
    model: Optional[str] = Query(None, description="Force specific model")
):
    """
    Content-based similar tour recommendations.

    Returns tours similar to the given tour based on:
    - Destination similarity
    - Price range similarity
    - Duration similarity
    - Description content similarity
    """
    try:
        if model == "deepfm":
            recommender = getattr(request.app.state, "deepfm_recommender", None)
            model_used = "deepfm"
        elif model == "hybrid":
            recommender = getattr(request.app.state, "recommender", None)
            model_used = "hybrid"
        else:
            recommender, model_used = get_recommender(request, prefer_deepfm=True)

        if not recommender:
            logger.warning("Recommender not initialized")
            return {"data": [], "model": None}

        logger.info(f"Getting similar tours for {tourId} using {model_used}")
        tours = await recommender.get_similar_tours(tourId, limit)

        return {"data": [format_tour_response(t) for t in tours], "model": model_used}

    except Exception as e:
        logger.error(f"Similar tours error: {e}", exc_info=True)
        return {"data": [], "model": None}


@router.get("/post-booking")
async def get_post_booking_recommendations(
    request: Request,
    tourId: str = Query(..., description="Just booked tour ID"),
    userId: Optional[str] = Query(None, description="User ID"),
    limit: int = Query(4, ge=1, le=10, description="Number of recommendations"),
    model: Optional[str] = Query(None, description="Force specific model")
):
    """
    Post-booking recommendations shown after successful booking.

    Combines:
    - Similar tours (content-based)
    - What others who booked this also booked (collaborative)
    - DeepFM personalization (if available)
    """
    try:
        if model == "deepfm":
            recommender = getattr(request.app.state, "deepfm_recommender", None)
            model_used = "deepfm"
        elif model == "hybrid":
            recommender = getattr(request.app.state, "recommender", None)
            model_used = "hybrid"
        else:
            recommender, model_used = get_recommender(request, prefer_deepfm=True)

        if not recommender:
            logger.warning("Recommender not initialized")
            return {"data": [], "model": None}

        logger.info(f"Getting post-booking recommendations for tour {tourId}, user {userId} using {model_used}")

        if model_used == "deepfm":
            tours = await recommender.get_post_booking_recommendations(tourId, userId, limit)
        else:
            tours = await recommender.get_post_booking_recommendations(tourId, userId, limit)

        # Fallback to similar tours if not enough results
        if len(tours) < limit:
            similar = await recommender.get_similar_tours(tourId, limit)
            existing_ids = {str(t.get("_id", "")) for t in tours}

            for s in similar:
                if str(s.get("_id", "")) not in existing_ids and len(tours) < limit:
                    tours.append(s)

        return {"data": [format_tour_response(t) for t in tours], "model": model_used}

    except Exception as e:
        logger.error(f"Post-booking recommendation error: {e}", exc_info=True)

        # Fallback to similar tours
        try:
            recommender, _ = get_recommender(request)
            if recommender:
                similar = await recommender.get_similar_tours(tourId, limit)
                return {"data": [format_tour_response(t) for t in similar], "model": "fallback"}
        except Exception:
            pass

        return {"data": [], "model": None}


@router.get("/model/info")
async def get_model_info(request: Request):
    """Get information about the recommendation models."""
    deepfm = getattr(request.app.state, "deepfm_recommender", None)
    hybrid = getattr(request.app.state, "recommender", None)

    info = {
        "default_model": "deepfm" if deepfm and deepfm.is_initialized else "hybrid",
        "deepfm": deepfm.get_model_info() if deepfm else {"status": "not_initialized"},
        "hybrid": {
            "status": "active" if hybrid else "not_initialized"
        }
    }

    return info


@router.post("/model/retrain")
async def retrain_model(
    request: Request,
    epochs: int = Query(10, ge=1, le=50, description="Number of training epochs")
):
    """Retrain the DeepFM model with current data."""
    deepfm = getattr(request.app.state, "deepfm_recommender", None)

    if not deepfm:
        raise HTTPException(status_code=503, detail="DeepFM recommender not available")

    logger.info(f"Starting model retraining with {epochs} epochs")
    result = await deepfm.retrain(epochs=epochs)

    return {
        "status": "success",
        "result": result
    }


@router.post("/track")
async def track_interaction(
    request: Request,
    data: TrackingRequest
):
    """
    Track user interaction với đầy đủ context.

    Request body:
    - userId: ID người dùng (optional cho anonymous)
    - tourId: ID tour
    - type: Loại tương tác (view, click, bookmark, share, booking, review)
    - value: Giá trị (1.0 mặc định, rating cho review)
    - source: Nguồn (homepage, similar, post_booking, search, direct)
    - model: Model đã dùng (deepfm, hybrid, popularity)
    - position: Vị trí trong danh sách (0-indexed)
    - sessionId: Session ID
    - deviceType: Loại thiết bị (mobile, desktop, tablet)
    - duration: Thời gian xem (giây)
    """
    interaction_service = getattr(request.app.state, "interaction_service", None)
    deepfm = getattr(request.app.state, "deepfm_recommender", None)

    result = {"status": "recorded", "tourId": data.tourId, "type": data.type}

    # 1. Lưu vào database
    if interaction_service:
        try:
            interaction_id = await interaction_service.track(
                user_id=data.userId,
                tour_id=data.tourId,
                interaction_type=data.type,
                value=data.value,
                source=data.source,
                model=data.model,
                position=data.position,
                session_id=data.sessionId,
                device_type=data.deviceType,
                duration=data.duration
            )
            result["interactionId"] = interaction_id
        except Exception as e:
            logger.error(f"Failed to save interaction: {e}")
            result["db_error"] = str(e)

    # 2. Online learning cho DeepFM (chỉ với các interaction có ý nghĩa)
    if deepfm and data.userId and data.type in ["click", "booking", "review"]:
        try:
            label_map = {
                "booking": 1.0,
                "review": data.value / 5.0,  # Normalize rating
                "click": 0.7,
            }
            label = label_map.get(data.type, 0.5)

            await deepfm.record_interaction(
                data.userId, data.tourId, data.type, label
            )
            result["model_updated"] = True
        except Exception as e:
            logger.error(f"Failed to update model: {e}")

    return result


@router.post("/interaction")
async def record_interaction(
    request: Request,
    userId: str = Query(..., description="User ID"),
    tourId: str = Query(..., description="Tour ID"),
    interaction_type: str = Query(..., description="Type: booking, review, click, view"),
    value: float = Query(1.0, description="Interaction value (e.g., rating)")
):
    """
    [DEPRECATED] Use POST /track instead.
    Record user interaction for online learning.
    """
    deepfm = getattr(request.app.state, "deepfm_recommender", None)
    interaction_service = getattr(request.app.state, "interaction_service", None)

    # Save to database
    if interaction_service:
        await interaction_service.track(
            user_id=userId,
            tour_id=tourId,
            interaction_type=interaction_type,
            value=value
        )

    # Update model
    if deepfm:
        await deepfm.record_interaction(userId, tourId, interaction_type, value)

    return {
        "status": "recorded",
        "userId": userId,
        "tourId": tourId,
        "type": interaction_type
    }


@router.get("/compare")
async def compare_recommendations(
    request: Request,
    userId: Optional[str] = Query(None, description="User ID"),
    tourId: Optional[str] = Query(None, description="Tour ID for similar"),
    limit: int = Query(4, ge=1, le=10, description="Number of recommendations")
):
    """
    Compare recommendations from different models side by side.

    Useful for A/B testing and model evaluation.
    """
    deepfm = getattr(request.app.state, "deepfm_recommender", None)
    hybrid = getattr(request.app.state, "recommender", None)
    tour_service = getattr(request.app.state, "tour_service", None)

    result = {
        "homepage": {},
        "similar": {}
    }

    # Compare homepage recommendations
    if userId:
        if deepfm and deepfm.is_initialized:
            try:
                deepfm_recs = await deepfm.get_homepage_recommendations(userId, limit)
                result["homepage"]["deepfm"] = [format_tour_response(t) for t in deepfm_recs]
            except Exception as e:
                result["homepage"]["deepfm"] = {"error": str(e)}

        if hybrid:
            try:
                hybrid_recs = await hybrid.get_homepage_recommendations(userId, limit)
                result["homepage"]["hybrid"] = [format_tour_response(t) for t in hybrid_recs]
            except Exception as e:
                result["homepage"]["hybrid"] = {"error": str(e)}
    else:
        if tour_service:
            popular = await tour_service.get_popular_tours(limit)
            result["homepage"]["popularity"] = [format_tour_response(t) for t in popular]

    # Compare similar tours
    if tourId:
        if deepfm and deepfm.is_initialized:
            try:
                deepfm_similar = await deepfm.get_similar_tours(tourId, limit)
                result["similar"]["deepfm"] = [format_tour_response(t) for t in deepfm_similar]
            except Exception as e:
                result["similar"]["deepfm"] = {"error": str(e)}

        if hybrid:
            try:
                hybrid_similar = await hybrid.get_similar_tours(tourId, limit)
                result["similar"]["hybrid"] = [format_tour_response(t) for t in hybrid_similar]
            except Exception as e:
                result["similar"]["hybrid"] = {"error": str(e)}

    return result


# ==================== Analytics Endpoints ====================

@router.get("/analytics/metrics")
async def get_recommendation_metrics(
    request: Request,
    model: Optional[str] = Query(None, description="Filter by model"),
    source: Optional[str] = Query(None, description="Filter by source"),
    days: int = Query(7, ge=1, le=90, description="Number of days")
):
    """
    Lấy metrics của recommendation system.

    Returns:
    - impressions: Số lần hiển thị
    - clicks: Số click
    - ctr: Click-through rate (%)
    - bookings: Số booking từ recommendations
    - conversion_rate: Tỷ lệ chuyển đổi (%)
    """
    interaction_service = getattr(request.app.state, "interaction_service", None)

    if not interaction_service:
        raise HTTPException(status_code=503, detail="Interaction service not available")

    metrics = await interaction_service.get_recommendation_metrics(
        model=model,
        source=source,
        days=days
    )

    return {"days": days, "metrics": metrics}


@router.get("/analytics/ab-test")
async def get_ab_test_results(
    request: Request,
    days: int = Query(7, ge=1, le=90, description="Number of days")
):
    """
    So sánh hiệu quả giữa các models (A/B testing).

    Returns comparison of:
    - DeepFM vs Hybrid vs Popularity
    - CTR, Conversion Rate for each model
    """
    interaction_service = getattr(request.app.state, "interaction_service", None)

    if not interaction_service:
        raise HTTPException(status_code=503, detail="Interaction service not available")

    results = await interaction_service.get_ab_test_results(days=days)

    return {"days": days, "results": results}


@router.get("/analytics/daily")
async def get_daily_stats(
    request: Request,
    days: int = Query(7, ge=1, le=30, description="Number of days")
):
    """
    Thống kê interactions theo ngày.

    Returns daily breakdown of:
    - views, clicks, bookings, reviews per day
    """
    interaction_service = getattr(request.app.state, "interaction_service", None)

    if not interaction_service:
        raise HTTPException(status_code=503, detail="Interaction service not available")

    stats = await interaction_service.get_daily_stats(days=days)

    return {"days": days, "data": stats}


@router.get("/analytics/tour/{tour_id}")
async def get_tour_analytics(
    request: Request,
    tour_id: str,
    days: int = Query(30, ge=1, le=90, description="Number of days")
):
    """
    Thống kê interactions của một tour cụ thể.
    """
    interaction_service = getattr(request.app.state, "interaction_service", None)

    if not interaction_service:
        raise HTTPException(status_code=503, detail="Interaction service not available")

    stats = await interaction_service.get_tour_interactions(tour_id, days=days)

    return {"tourId": tour_id, "days": days, "interactions": stats}


@router.get("/analytics/user/{user_id}")
async def get_user_history(
    request: Request,
    user_id: str,
    types: Optional[str] = Query(None, description="Comma-separated types: view,click,booking"),
    days: int = Query(30, ge=1, le=90, description="Number of days"),
    limit: int = Query(50, ge=1, le=200, description="Max results")
):
    """
    Lấy lịch sử tương tác của user.
    """
    interaction_service = getattr(request.app.state, "interaction_service", None)

    if not interaction_service:
        raise HTTPException(status_code=503, detail="Interaction service not available")

    interaction_types = types.split(",") if types else None

    history = await interaction_service.get_user_history(
        user_id=user_id,
        interaction_types=interaction_types,
        limit=limit,
        days=days
    )

    # Format response
    formatted = []
    for item in history:
        formatted.append({
            "id": str(item["_id"]),
            "tourId": str(item["tourId"]),
            "type": item["type"],
            "value": item.get("value"),
            "source": item.get("source"),
            "model": item.get("model"),
            "createdAt": item["createdAt"].isoformat() if item.get("createdAt") else None
        })

    return {"userId": user_id, "count": len(formatted), "history": formatted}
