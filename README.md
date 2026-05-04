# Tour Recommendation Service

Dịch vụ gợi ý tour cá nhân hóa sử dụng thuật toán Hybrid (Collaborative Filtering + Content-Based).

## Cài đặt

### Yêu cầu
- Python 3.11+
- MongoDB

### Setup

```bash
cd recommendation-service

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Copy và cấu hình .env
cp .env.example .env
# Sửa MONGODB_URI trong .env
```

### Chạy service

```bash
# Development
uvicorn app.main:app --reload --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Chạy với Docker

```bash
docker-compose up --build
```

## API Endpoints

| Endpoint | Mô tả |
|----------|-------|
| `GET /health` | Health check |
| `GET /recommend/homepage?userId=&limit=6` | Gợi ý trang chủ cá nhân hóa |
| `GET /recommend/similar?tourId=&limit=4` | Tour tương tự |
| `GET /recommend/post-booking?tourId=&userId=&limit=4` | Gợi ý sau khi đặt tour |

### Response format

```json
{
  "data": [
    {
      "_id": "...",
      "title": "Tour name",
      "destination": "Da Nang",
      "priceAdult": 3500000,
      "images": ["..."]
    }
  ]
}
```

## Thuật toán

### Hybrid Algorithm

Kết hợp 3 phương pháp với trọng số động dựa trên số lượng booking của user:

| User Level | Bookings | CF | CB | Popularity |
|------------|----------|----|----|------------|
| Cold Start | 0 | 0% | 30% | 70% |
| New | 1-2 | 20% | 30% | 50% |
| Regular | 3-10 | 40% | 40% | 20% |
| Active | >10 | 50% | 40% | 10% |

### Công thức

```
score(user, tour) = α × CF_score + β × CB_score + γ × popularity_score
```

- **CF_score**: Dựa trên users tương tự đã book tour đó
- **CB_score**: Cosine similarity giữa user profile và tour features
- **popularity_score**: 0.6 × norm(booking_count) + 0.4 × norm(avg_rating)

## Testing

```bash
# Chạy tests
pytest tests/ -v

# Chạy với coverage
pytest tests/ --cov=app --cov-report=html
```

## Cấu trúc thư mục

```
recommendation-service/
├── app/
│   ├── main.py                      # FastAPI entry point
│   ├── config/
│   │   ├── settings.py              # Environment config
│   │   └── database.py              # MongoDB connection
│   ├── models/
│   │   └── schemas.py               # Pydantic models
│   ├── services/
│   │   ├── tour_service.py          # Data access
│   │   ├── feature_engineering.py   # Feature extraction
│   │   ├── collaborative.py         # CF algorithm
│   │   ├── content_based.py         # CB algorithm
│   │   └── hybrid_recommender.py    # Main recommender
│   ├── api/
│   │   └── recommendations.py       # API endpoints
│   └── utils/
│       └── cache.py                 # Caching
├── tests/
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```
