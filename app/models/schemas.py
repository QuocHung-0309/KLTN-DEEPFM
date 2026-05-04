from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime


class TourResponse(BaseModel):
    """Tour data returned by recommendation API."""
    id: str = Field(alias="_id")
    title: str
    destination: str
    destinationSlug: Optional[str] = None
    priceAdult: int = 0
    priceChild: Optional[int] = None
    time: Optional[str] = None
    description: Optional[str] = None
    images: List[str] = []

    class Config:
        populate_by_name = True


class RecommendationResponse(BaseModel):
    """Standard response format for all recommendation endpoints."""
    data: List[TourResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str


class TourFeatures(BaseModel):
    """Internal model for tour feature vector."""
    tour_id: str
    destination: str
    destination_slug: str
    price_normalized: float
    duration_days: int
    avg_rating: float
    booking_count: int
    feature_vector: Optional[List[float]] = None


class UserProfile(BaseModel):
    """Internal model for user preference profile."""
    user_id: str
    booked_tour_ids: List[str]
    reviewed_tour_ids: List[str]
    preferred_destinations: List[str]
    avg_price_preference: float
    total_bookings: int
    user_level: str  # cold, new, regular, active
