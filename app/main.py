from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os

from app.config.settings import settings
from app.config.database import connect_to_mongo, close_mongo_connection, get_database

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    logger.info("Starting recommendation service...")
    await connect_to_mongo(settings.MONGODB_URI, settings.DB_NAME)
    db = get_database()

    # Import services
    from app.services.tour_service import TourService
    from app.services.feature_engineering import TourFeatureExtractor, TourSimilarityCalculator
    from app.services.collaborative import CollaborativeFilter
    from app.services.content_based import ContentBasedFilter
    from app.services.hybrid_recommender import HybridRecommender
    from app.services.interaction_service import InteractionService

    tour_service = TourService()

    # Initialize interaction service for tracking
    logger.info("Initializing interaction service...")
    interaction_service = InteractionService(db)
    await interaction_service.ensure_indexes()
    app.state.interaction_service = interaction_service

    # Build feature extractor
    logger.info("Building feature extractor...")
    feature_extractor = TourFeatureExtractor()
    tours = await tour_service.get_all_tours()

    if tours:
        feature_extractor.fit(tours)

        # Build similarity calculator
        logger.info("Building similarity matrix...")
        similarity_calculator = TourSimilarityCalculator(feature_extractor)
        similarity_calculator.build_similarity_matrix(tours)

        # Build collaborative filter
        logger.info("Building collaborative filter...")
        cf = CollaborativeFilter()
        bookings = await tour_service.get_all_bookings()
        reviews = await tour_service.get_all_reviews()
        await cf.build_model(bookings, reviews)

        # Build content-based filter
        cb = ContentBasedFilter(feature_extractor, similarity_calculator)

        # Initialize hybrid recommender
        logger.info("Initializing hybrid recommender...")
        hybrid_recommender = HybridRecommender(cf, cb, tour_service)
        await hybrid_recommender.initialize()

        # Initialize DeepFM recommender
        logger.info("Initializing DeepFM recommender...")
        try:
            from app.services.deepfm_recommender import DeepFMRecommenderService
            deepfm_recommender = DeepFMRecommenderService(
                tour_service=tour_service,
                model_path="models/deepfm"
            )
            await deepfm_recommender.initialize(db)
            app.state.deepfm_recommender = deepfm_recommender
            logger.info("DeepFM recommender initialized successfully!")
        except Exception as e:
            logger.warning(f"DeepFM initialization failed (falling back to Hybrid): {e}")
            app.state.deepfm_recommender = None

        # Store in app state
        app.state.recommender = hybrid_recommender
        app.state.tour_service = tour_service
        app.state.similarity_calculator = similarity_calculator
    else:
        logger.warning("No tours found in database")
        app.state.recommender = None
        app.state.deepfm_recommender = None
        app.state.tour_service = tour_service
        app.state.similarity_calculator = None

    logger.info("Recommendation service ready!")

    yield

    # Shutdown
    await close_mongo_connection()
    logger.info("Recommendation service stopped.")


app = FastAPI(
    title="Tour Recommendation Service",
    description="Personalized tour recommendations using DeepFM + Hybrid Algorithm",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    deepfm_status = "active" if hasattr(app.state, 'deepfm_recommender') and app.state.deepfm_recommender else "inactive"
    hybrid_status = "active" if hasattr(app.state, 'recommender') and app.state.recommender else "inactive"

    return {
        "status": "healthy",
        "service": "recommendation",
        "models": {
            "deepfm": deepfm_status,
            "hybrid": hybrid_status
        }
    }


# Import and include API router
from app.api.recommendations import router as recommendations_router
app.include_router(recommendations_router, prefix="/recommend", tags=["Recommendations"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
