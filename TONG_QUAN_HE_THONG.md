# TỔNG QUAN HỆ THỐNG GỢI Ý TOUR DU LỊCH — TRAVELA

> Khóa luận tốt nghiệp — Hệ thống gợi ý tour cá nhân hóa  
> Cập nhật: 26/05/2026

---

## MỤC LỤC

1. [Giới thiệu](#1-giới-thiệu)
2. [Kiến trúc tổng thể](#2-kiến-trúc-tổng-thể)
3. [Dữ liệu](#3-dữ-liệu)
4. [Thuật toán gợi ý](#4-thuật-toán-gợi-ý)
   - 4.1 [Feature Engineering](#41-feature-engineering)
   - 4.2 [Collaborative Filtering — CF](#42-collaborative-filtering-cf)
   - 4.3 [Content-Based Filtering — CB](#43-content-based-filtering-cb)
   - 4.4 [Hybrid Recommender](#44-hybrid-recommender)
   - 4.5 [DeepFM (LibRecommender)](#45-deepfm--librerecommender)
   - 4.6 [Pipeline Tổng Hợp](#46-pipeline-tổng-hợp-hybrid--deepfm-re-rank)
5. [Xử lý Cold-Start](#5-xử-lý-cold-start)
6. [API Endpoints](#6-api-endpoints)
7. [Tracking & Analytics](#7-tracking--analytics)
8. [Đánh giá & Metrics](#8-đánh-giá--metrics)
9. [Khởi động hệ thống](#9-khởi-động-hệ-thống)
10. [Cấu trúc thư mục](#10-cấu-trúc-thư-mục)
11. [Dependencies](#11-dependencies-chính)

---

## 1. Giới thiệu

Hệ thống gợi ý tour du lịch cá nhân hóa được xây dựng dưới dạng **microservice độc lập** (FastAPI / Python 3.9), tích hợp với backend Node.js và frontend Next.js của nền tảng đặt tour **Travela**.

### Mục tiêu

- Gợi ý tour **phù hợp sở thích và lịch sử** của từng người dùng
- Xử lý tốt bài toán **cold-start** (người dùng mới chưa có lịch sử)
- Kết hợp nhiều phương pháp (**Hybrid**) để tận dụng ưu điểm từng thuật toán
- Sử dụng **DeepFM** (mạng nơ-ron sâu) để re-rank kết quả

### Loại gợi ý

| Loại | Endpoint | Mô tả |
|------|----------|--------|
| Trang chủ | `GET /recommend/homepage` | Gợi ý cá nhân hóa (Hybrid + DeepFM) |
| Tour tương tự | `GET /recommend/similar` | Cosine similarity theo features |
| Sau đặt tour | `GET /recommend/post-booking` | CB similar + CF co-purchase |
| Phổ biến | Fallback anonymous | Popularity score |

---

## 2. Kiến trúc tổng thể

```
┌──────────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser)                          │
└──────────────────────────────┬───────────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼───────────────────────────────────┐
│                   Frontend  (Next.js · port 3000)                │
└──────────────────────────────┬───────────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼───────────────────────────────────┐
│                   Backend  (Node.js · port 3001)                  │
│  - Xác thực JWT, quản lý tour/booking/review                     │
│  - Proxy request sang Recommendation Service                     │
└──────────┬───────────────────────────────┬───────────────────────┘
           │ MongoDB Atlas                 │ HTTP
           │               ┌──────────────▼──────────────────────┐
           │               │   Recommendation Service             │
           │               │   FastAPI · Python 3.9 · port 8000   │
           │               │                                      │
           │               │  ┌───────────────────────────────┐  │
           │               │  │         API Layer             │  │
           │               │  │  /homepage /similar /track    │  │
           │               │  └──────────────┬────────────────┘  │
           │               │                 │                    │
           │               │  ┌──────────────▼────────────────┐  │
           │               │  │   DeepFMRecommenderService    │  │
           │               │  │  (Hybrid primary + DeepFM     │  │
           │               │  │   re-rank 30%)                │  │
           │               │  └───┬───────────────────────────┘  │
           │               │      │                               │
           │               │  ┌───▼──────────┐  ┌─────────────┐ │
           │               │  │  Hybrid      │  │  DeepFM     │ │
           │               │  │  CF+CB+Pop   │  │  (LibReco)  │ │
           │               │  └──────────────┘  └─────────────┘ │
           │               └──────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────┐
│                  MongoDB Atlas  (DB: travela)                    │
│  tbl_tours · tbl_booking · tbl_reviews · tbl_users              │
│  tbl_tour_departures · tbl_user_interactions                    │
└─────────────────────────────────────────────────────────────────┘
```

### Khởi động tại startup (`app/main.py`)

```
1. Kết nối MongoDB Atlas
2. Load tất cả tours từ DB
3. Fit TourFeatureExtractor  → vector 81 chiều / tour
4. Build TourSimilarityMatrix → ma trận cosine N×N (pre-compute)
5. Build CollaborativeFilter  → user-item rating matrix
6. Build ContentBasedFilter   → user profile builder
7. Init HybridRecommender     → CF + CB + Popularity với dynamic weights
8. Init DeepFMRecommender     → load weights từ models/deepfm_libreco/
9. API sẵn sàng nhận request
```

---

## 3. Dữ liệu

### 3.1 Nguồn dữ liệu

Toàn bộ dữ liệu lưu trên **MongoDB Atlas** (cloud), database `travela`.

| Collection | Mô tả | Số bản ghi (hiện tại) |
|------------|--------|----------------------|
| `tbl_tours` | Thông tin tour | 43 |
| `tbl_tour_departures` | Lịch khởi hành | 120 |
| `tbl_booking` | Đơn đặt tour | 651 |
| `tbl_reviews` | Đánh giá (1–5 sao) | 309 |
| `tbl_users` | Người dùng | 71 |
| `tbl_user_interactions` | View / Click / Bookmark / Share | 1 383 |

### 3.2 Schema các collection chính

**tbl_tours**
```
_id, title, destination, priceAdult, priceChild, salePrice,
time, description, images, quantity, destinationSlug
```

**tbl_booking**
```
_id, userId, tourDepartureId → (tbl_tour_departures.tourId),
bookingStatus [pending | confirmed | completed | cancelled],
numAdults, numChildren, totalPrice, createdAt
```
> Lưu ý: booking không có `tourId` trực tiếp — phải join qua `tourDepartureId → tbl_tour_departures.tourId`

**tbl_reviews**
```
_id, userId, tourId, rating (1–5), comment, createdAt
```

**tbl_user_interactions**
```
_id, userId, tourId, type [view | click | bookmark | share],
source [homepage | similar | post_booking | search | direct],
model [deepfm | hybrid | popularity], position, deviceType, createdAt
```

### 3.3 Gán nhãn (Label Assignment) cho DeepFM

Mỗi cặp `(user, tour)` được gán nhãn từ **tất cả nguồn tương tác**, lấy **max** làm label cuối:

| Nguồn | Nhãn |
|-------|------|
| Review | `rating` (1 → 5) |
| Booking completed | 5.0 |
| Booking confirmed | 4.0 |
| Booking pending | 3.0 |
| Interaction bookmark | 4.0 |
| Interaction share | 3.5 |
| Interaction click | 2.0 |
| Interaction view | 1.0 |

```
label(u, t) = max(review_rating, booking_weight, interaction_weight)
```

### 3.4 Thống kê dữ liệu training

| Thống kê | Giá trị |
|---------|---------|
| Unique users có tương tác | 71 |
| Unique tours có tương tác | 46 |
| Số cặp (user, tour) sau aggregate | ~402 |
| Mật độ ma trận | ~12.3% |
| Điểm đến (destinations) | 29 |

---

## 4. Thuật toán gợi ý

### 4.1 Feature Engineering

**File:** `app/services/feature_engineering.py`

Mỗi tour được biểu diễn bằng **vector đặc trưng 81 chiều**:

```
features(tour) = [destination_onehot(29) | price_norm(1) | duration_norm(1) | tfidf_desc(50)]
```

| Thành phần | Số chiều | Phương pháp |
|------------|----------|-------------|
| Destination one-hot | 29 | Mỗi tỉnh/thành = 1 bit |
| Price norm | 1 | MinMaxScaler → [0, 1] |
| Duration norm | 1 | Số ngày / 10, clip tại 1.0 |
| TF-IDF description | 50 | Top 50 từ quan trọng nhất từ mô tả tour |
| **Tổng** | **81** | |

**Parse thời gian tour** (VD: "3 ngày 2 đêm", "4N3D"):
```python
re.search(r"(\d+)\s*ng[aà]y", time_lower)   # "3 ngay"
re.search(r"(\d+)\s*n", time_lower)           # "4N"
```

**Ma trận Similarity** (`TourSimilarityCalculator`):
```
sim(i, j) = cosine(features(i), features(j))
           = features(i) · features(j)
             ─────────────────────────────
             ||features(i)|| × ||features(j)||
```
Pre-compute **1 lần lúc startup**, lưu ma trận N×N → tra cứu O(1).

---

### 4.2 Collaborative Filtering (CF)

**File:** `app/services/collaborative.py`  
**Phương pháp:** User-Based Collaborative Filtering

#### Bước 1 — Xây dựng ma trận Rating

```
rating(user, tour) = max(booking_weight=3.0, review_rating=1..5)
```

Dữ liệu đầu vào: `tbl_booking` (join qua departure → tourId) + `tbl_reviews`.

#### Bước 2 — Độ tương đồng giữa hai users (Cosine Similarity)

```
                  Σ_{t ∈ T_u1 ∩ T_u2}  r(u1, t) × r(u2, t)
sim(u1, u2) = ─────────────────────────────────────────────────
                     ||r(u1)||  ×  ||r(u2)||
```

- `T_u1 ∩ T_u2`: tập tour cả hai đều tương tác
- Ngưỡng tối thiểu: `sim ≥ 0.1`, tối đa **20 láng giềng** (N-nearest neighbors)

#### Bước 3 — Dự đoán điểm cho tour chưa tương tác

```
                  Σ_{v ∈ N(u)}  sim(u, v) × r(v, t)
pred(u, t) = ─────────────────────────────────────────
                   Σ_{v ∈ N(u)}  |sim(u, v)|
```

#### Co-purchase (dùng cho post-booking)

```
copurchase(t_booked, t_other) = count(users đã book cả t_booked VÀ t_other)
```

---

### 4.3 Content-Based Filtering (CB)

**File:** `app/services/content_based.py`

#### Bước 1 — Xây dựng User Profile

```
                Σ_{t ∈ history(u)}  weight(u, t) × features(t)
profile(u) = ────────────────────────────────────────────────────
                         Σ_{t} weight(u, t)

  weight khi booking = 3.0
  weight khi review  = rating (1..5)
```

Profile được **normalize** về unit vector.

#### Bước 2 — Tính điểm cho tour

```
cb_score(u, t) = cosine(profile(u), features(t))
```

#### Tour tương tự

Dùng `TourSimilarityCalculator.get_similar_tours(tour_id, top_k)` — tra ma trận pre-computed.

---

### 4.4 Hybrid Recommender

**File:** `app/services/hybrid_recommender.py`

#### Công thức kết hợp

```
score(u, t) = α × CF_score(u, t) + β × CB_score(u, t) + γ × Popularity(t)
```

#### Trọng số động theo mức hoạt động user

| Mức | Điều kiện | α (CF) | β (CB) | γ (Pop) | Lý do |
|-----|-----------|--------|--------|---------|-------|
| Cold Start | 0 booking | 0.0 | 0.3 | **0.7** | Chưa có lịch sử |
| New | 1–2 booking | 0.2 | 0.3 | **0.5** | Ít dữ liệu |
| Regular | 3–10 booking | 0.4 | 0.4 | 0.2 | Cân bằng |
| Active | >10 booking | **0.5** | 0.4 | 0.1 | Đủ dữ liệu CF |

#### Popularity Score

```
popularity(t) = 0.6 × norm_booking(t) + 0.4 × norm_rating(t)

  norm_booking(t) = booking_count(t) / max_booking_count
  norm_rating(t)  = avg_rating(t)   / 5.0
```

#### Diversity Filter

Sau khi sort điểm, lọc để **tối đa 2 tour / điểm đến** → tăng đa dạng kết quả.

#### Luồng xử lý Homepage

```
Request (userId, limit=6)
         │
         ▼
Lấy lịch sử booking + review của user
         │
         ▼
Xác định user level → chọn (α, β, γ)
         │
         ├── [α > 0] CF → predict tất cả tour chưa book → normalize
         ├── [β > 0] CB → build profile → cosine similarity → normalize
         └── [γ > 0] Popularity → pre-computed scores
         │
         ▼
score = α·CF + β·CB + γ·Pop
         │
         ▼
Sort giảm dần → Diversity filter (max 2/điểm đến) → Top-K
```

#### Post-Booking Recommendations

```
score(t) = 0.6 × CB_similar(t_booked, t) + 0.4 × CoPurchase(t_booked, t)
```

---

### 4.5 DeepFM — LibRecommender

**Files:** `app/services/deepfm_model.py`  
**Thư viện:** [LibRecommender v1.2.1](https://librecommender.readthedocs.io/en/v1.2.1/)  
**Backend:** TensorFlow 2.15 (TF1-compat graph mode)  
**Model lưu tại:** `models/deepfm_libreco/`

#### Kiến trúc DeepFM

DeepFM kết hợp **FM (Factorization Machine)** và **Deep Neural Network** trong cùng một model, chia sẻ chung tầng Embedding:

```
Features đầu vào:
  [user_id, tour_id, destination, price_bucket, duration_bucket]
                    │
        ┌───────────▼───────────┐
        │    Embedding Layer     │
        │  (mỗi feature → 16D)  │
        └───────┬───────────────┘
                │
        ┌───────┴───────────────────────────────┐
        │                                       │
┌───────▼──────┐                    ┌──────────▼──────────────────┐
│   FM Part    │                    │       Deep Part              │
│              │                    │  Dense(128) → BN → Dropout  │
│  1st order:  │                    │  Dense(64)  → BN → Dropout  │
│  Σ w_i·x_i  │                    │  Dense(32)  → BN → Dropout  │
│              │                    │  Dense(1)                    │
│  2nd order:  │                    └──────────┬──────────────────┘
│  ½Σ(v_i·x_i)│                               │
│    inner     │                               │
│    product   │                               │
└───────┬──────┘                               │
        │                                      │
        └──────────────────┬───────────────────┘
                           │
                    ┌──────▼──────┐
                    │   Sigmoid   │  → score ∈ [0, 1]
                    └─────────────┘
```

**Tổng params: 24,024** (embedding: 2,946 | network: 21,078)

#### Input Features cho DeepFM

| Feature | Encoding | Vai trò |
|---------|----------|---------|
| `destination` | Sparse categorical (29 giá trị) | Item feature |
| `price_bucket` | Sparse int (0–9, chia theo quantile 10%) | Item feature |
| `duration_bucket` | Sparse int (0–6, = số ngày − 1, clip 6) | Item feature |
| user_id | Tự động encode bởi LibRecommender | User ID |
| tour_id | Tự động encode bởi LibRecommender | Item ID |

**Price bucket** — chia theo phân vị giá của tập tour:
```
B0: ≤ 1.0M  B1: ≤ 2.0M  B2: ≤ 2.4M  B3: ≤ 3.2M  B4: ≤ 3.5M
B5: ≤ 3.8M  B6: ≤ 4.2M  B7: ≤ 4.8M  B8: ≤ 5.5M  B9: > 5.5M
```

**Duration bucket:**
```
0 = 1 ngày   1 = 2 ngày   2 = 3 ngày   3 = 4 ngày
4 = 5 ngày   5 = 6 ngày   6 = 7+ ngày
```

#### Cấu hình huấn luyện

```python
DeepFM(
    task          = "ranking",          # implicit feedback ranking
    loss_type     = "cross_entropy",
    embed_size    = 16,
    n_epochs      = 20,
    lr            = 1e-3,
    batch_size    = 256,
    hidden_units  = (128, 64, 32),
    use_bn        = True,
    dropout_rate  = 0.2,
    sampler       = "random",           # negative sampling
    num_neg       = 1,                  # 1 negative / 1 positive
)
```

#### Split dữ liệu

```
Toàn bộ: ~813 samples (71 users × 42 tours)
Train: 90% (~730 samples)
Eval:  10% (~83 samples)
```

#### Output

```python
deepfm.predict(user_id, [tour_id_1, ...])   → [(tour_id, score), ...]  score ∈ [0,1]
deepfm.recommend_user(user_id, n_rec=10)    → [tour_id_1, ...]  đã loại tour đã tương tác
```

---

### 4.6 Pipeline Tổng Hợp: Hybrid + DeepFM Re-rank

**File:** `app/services/deepfm_recommender.py`

#### Lý do chọn kiến trúc này

Với dữ liệu nhỏ (71 users, 813 samples), DeepFM đơn thuần đạt ROC-AUC ≈ 0.54 (gần random). Thay vì bỏ DeepFM, hệ thống dùng **Hybrid làm primary** (70%) và **DeepFM làm re-ranking signal** (30%).

#### Luồng xử lý

```
Request (userId, limit=6)
         │
         ▼
HybridRecommender.get_homepage_recommendations(userId, limit × 4)
         │  → candidate pool (24 tour)
         ▼
DeepFM.predict(userId, [candidate_tour_ids])
         │  → DeepFM score ∈ [0,1] cho từng tour
         ▼
Normalize DeepFM scores về [0,1]
         │
         ▼
Blend scores:
  final_score = 0.7 × Hybrid_score + 0.3 × DeepFM_norm_score
         │
         ▼
Sort giảm dần → Diversity filter (max 2/điểm đến) → Top-6
```

#### Công thức blend

```
final_score(u, t) = 0.7 × Hybrid_score(u, t)
                  + 0.3 × normalize(DeepFM_score(u, t))

  normalize(s) = (s - s_min) / (s_max - s_min)  — trên tập candidates
```

#### Fallback

```
DeepFM initialized?  ─No──► Hybrid only (không re-rank)
       │Yes
       ▼
DeepFM trained?  ─No──► Hybrid only
       │Yes
       ▼
DeepFM.predict() thành công?  ─Fail──► Hybrid only
       │Success
       ▼
Blend 70/30 → Diversity filter → Kết quả
```

#### Khi nào tăng DeepFM weight?

| Ngưỡng | DeepFM weight | Ghi chú |
|--------|---------------|---------|
| < 500 users (hiện tại) | 30% | Hybrid dominant |
| 500–2000 users | 40% | Tăng dần |
| > 2000 users + AUC > 0.75 | 50% | DeepFM chủ đạo |

---

## 5. Xử lý Cold-Start

### User Cold-Start

| Tình huống | Chiến lược |
|------------|------------|
| User không login | Trả Popularity (top phổ biến), không cá nhân hóa |
| User mới, 0 booking | CF=0%, CB=30%, Pop=70% |
| User có 1–2 booking | CF=20%, CB=30%, Pop=50% |
| User trong DeepFM nhưng chưa có embedding | `cold_start="average"` — dùng embedding trung bình |

### Item Cold-Start

Tour mới chưa có booking/review vẫn được gợi ý qua:
- **CB score** từ features (destination, price, duration)
- **Similarity matrix** — được tính từ features → xuất hiện trong `/similar`

---

## 6. API Endpoints

Base URL: `http://localhost:8000`  
Swagger UI: `http://localhost:8000/docs`

### 6.1 Gợi ý trang chủ

```
GET /recommend/homepage?userId=<id>&limit=6
```

| Param | Mô tả | Mặc định |
|-------|--------|---------|
| `userId` | MongoDB ObjectId string (không bắt buộc) | null → popular |
| `limit` | Số tour trả về (tối đa 20) | 6 |

**Response:**
```json
{
  "data": [
    {
      "_id": "...",
      "title": "Tour Phú Quốc 4N3D",
      "destination": "Phú Quốc",
      "priceAdult": 4500000,
      "_score": 0.712,
      "upcomingDepartures": [...]
    }
  ],
  "model": "deepfm_hybrid"
}
```

### 6.2 Tour tương tự

```
GET /recommend/similar?tourId=<id>&limit=4
```

Dùng **pre-computed cosine similarity matrix** → O(N), rất nhanh.

### 6.3 Gợi ý sau đặt tour

```
GET /recommend/post-booking?tourId=<id>&userId=<id>&limit=4
```

`score = 0.6 × CB_similar + 0.4 × CF_co_purchase`

### 6.4 Track tương tác

```
POST /recommend/track
{
  "userId": "...",
  "tourId": "...",
  "type": "click",       // view | click | bookmark | share
  "source": "homepage",  // homepage | similar | post_booking
  "model": "deepfm",
  "position": 2,
  "deviceType": "mobile"
}
```

### 6.5 Analytics

```
GET /recommend/analytics/metrics?days=7
GET /recommend/analytics/ab-test?days=7
GET /recommend/analytics/daily?days=7
GET /recommend/analytics/tour/{tour_id}?days=30
GET /recommend/analytics/user/{user_id}?days=30
```

### 6.6 Model Management

```
GET  /recommend/model/info
POST /recommend/model/retrain?epochs=10
GET  /recommend/compare?userId=<id>&tourId=<id>&limit=4
GET  /health
```

---

## 7. Tracking & Analytics

**File:** `app/services/interaction_service.py`

### Luồng tracking

```
User tương tác (click / view / bookmark / share)
         │
         ▼
POST /recommend/track
         │
         ├──► Lưu vào tbl_user_interactions (MongoDB)
         │    (userId, tourId, type, source, model, position, ...)
         │
         └──► Nếu DeepFM initialized
              → online_update(): tích lũy interaction
              → Retrain khi đủ 100 interactions mới
```

### Metrics theo dõi

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **CTR** | clicks / impressions | Tỷ lệ click |
| **Conversion** | bookings / clicks | Tỷ lệ đặt tour từ gợi ý |
| **Coverage** | unique_rec_tours / total_tours | % catalog được gợi ý |
| **Diversity** | avg unique destinations / list | Đa dạng điểm đến |
| **Personalization** | 1 − avg Jaccard(user_i, user_j) | Khác biệt giữa users |

---

## 8. Đánh giá & Metrics

### 8.1 DeepFM (LibRecommender)

| Chỉ số | Giá trị |
|--------|---------|
| Training samples (sau aggregate) | ~402 cặp (user, tour) |
| Users encode | 71 |
| Tours encode | 42 |
| Epochs | 20 |
| Embed size | 16 |
| Hidden units | (128, 64, 32) |
| Model params | 24,024 |

**Nhận xét:** Với dữ liệu nhỏ (71 users), DeepFM có xu hướng overfit. ROC-AUC đạt ~0.54 trên eval — gần random. Đây là lý do hệ thống dùng **Hybrid làm primary (70%)** thay vì để DeepFM làm standalone.

### 8.2 Hybrid Recommender

Đánh giá offline theo 3 chỉ số (n=15 users, top-6):

| Model | Coverage | Diversity | Personalization |
|-------|----------|-----------|-----------------|
| Popularity | thấp nhất | thấp | 0% (giống nhau) |
| Hybrid (CF+CB+Pop) | cao hơn | cao hơn | tốt |
| **Hybrid + DeepFM** | cao nhất | cao nhất | **tốt nhất** |

- **Coverage**: % tour catalog xuất hiện ít nhất 1 lần trong tập gợi ý
- **Diversity**: % unique destinations trong mỗi list gợi ý (avg)
- **Personalization**: 1 − avg Jaccard similarity giữa các user

### 8.3 Hạn chế & Hướng cải thiện

| Hạn chế | Nguyên nhân | Hướng cải thiện |
|---------|-------------|-----------------|
| Dữ liệu nhỏ (71 users) | Hệ thống còn mới | Mở rộng user base, thêm synthetic data |
| DeepFM AUC thấp (~0.54) | Ít training samples | Thêm implicit feedback (scroll, time-on-page) |
| Similarity matrix tĩnh | Build 1 lần lúc startup | Retrain định kỳ (cron job hàng ngày) |
| Không có temporal signal | Chưa dùng thời gian tương tác | Thêm recency weighting |

---

## 9. Khởi động hệ thống

Chạy **3 terminal song song**:

### Terminal 1 — Recommendation Service

```bash
cd ~/KLTN/recommendation-service
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Kiểm tra: `curl http://localhost:8000/health`

### Terminal 2 — Backend (Node.js)

```bash
cd ~/KLTN/TLCN-BE
npm run dev
```

### Terminal 3 — Frontend (Next.js)

```bash
cd ~/KLTN/TLCN-FE
npm run dev
```

Truy cập: `http://localhost:3000`

### Retrain DeepFM (từ notebook)

Mở `notebooks/he_thong_goi_y.ipynb`, kernel **Python (rec-service)**, chạy cell **11. Retrain DeepFM**.

Hoặc gọi API:
```bash
curl -X POST http://localhost:8000/recommend/model/retrain?epochs=20
```

---

## 10. Cấu trúc thư mục

```
recommendation-service/
│
├── app/
│   ├── main.py                      # FastAPI entry point, startup init
│   │
│   ├── api/
│   │   └── recommendations.py       # Tất cả API endpoints
│   │
│   ├── config/
│   │   ├── settings.py              # Env config (MONGODB_URI, PORT, DEBUG)
│   │   └── database.py              # MongoDB connection (Motor async)
│   │
│   ├── models/
│   │   └── schemas.py               # Pydantic request/response schemas
│   │
│   └── services/
│       ├── tour_service.py          # Data access layer (MongoDB queries)
│       ├── feature_engineering.py   # TourFeatureExtractor + SimilarityCalculator
│       ├── collaborative.py         # User-Based CF algorithm
│       ├── content_based.py         # Content-Based Filter + user profile
│       ├── hybrid_recommender.py    # Hybrid orchestrator (CF + CB + Pop)
│       ├── deepfm_model.py          # DeepFM via LibRecommender (train/predict)
│       ├── deepfm_recommender.py    # Pipeline service (Hybrid → DeepFM re-rank)
│       └── interaction_service.py   # Tracking & analytics queries
│
├── models/
│   └── deepfm_libreco/              # Trained DeepFM model (LibRecommender format)
│       ├── deepfm_tf_variables.npz  # Model weights
│       ├── deepfm_data_info.npz     # Feature encoding info
│       ├── deepfm_hyper_parameters.json
│       ├── deepfm_user_consumed.pkl # Items each user has interacted with
│       ├── deepfm_default_recs.npz  # Default recs for cold-start
│       └── meta.pkl                 # Encoders + training_history + metadata
│
├── notebooks/
│   ├── he_thong_goi_y.ipynb         # Demo toàn bộ pipeline (CHẠY NOTEBOOK NÀY)
│   ├── deepfm_travela.ipynb         # Quá trình train DeepFM step-by-step
│   └── svg/                         # Biểu đồ SVG cho báo cáo
│
├── .env                             # MONGODB_URI, PORT, DEBUG (không commit)
├── .env.example                     # Template cấu hình
├── .gitignore
├── requirements.txt                 # Python dependencies
├── Dockerfile
├── docker-compose.yml
└── TONG_QUAN_HE_THONG.md           # File này
```

---

## 11. Dependencies chính

| Package | Version | Dùng cho |
|---------|---------|----------|
| `fastapi` | 0.109+ | Web framework |
| `uvicorn` | 0.27+ | ASGI server |
| `motor` | 3.3+ | Async MongoDB driver |
| `librecommender` | **1.2.1** | DeepFM model (FM + Deep NN) |
| `tensorflow-macos` | **2.15.0** | Backend cho LibRecommender |
| `torch` | 2.x | PyTorch (LibRecommender dual-backend) |
| `numpy` | 1.26+ | Matrix operations |
| `scikit-learn` | 1.4+ | TF-IDF, cosine similarity, MinMaxScaler |
| `pandas` | 2.1+ | Data processing khi train |
| `pydantic-settings` | 2.x | Config validation |
| `python-dotenv` | 1.0+ | Load .env |

### Môi trường Python

```bash
# Venv chứa đủ TF + LibRecommender:
~/KLTN/recommendation-service/venv   (Python 3.9.6)

# Jupyter kernel đúng:
Python (rec-service)  hoặc  Travela DeepFM (Python 3.9)
```
