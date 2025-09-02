# Simple ML Project

A complete machine learning project for house price prediction with training, evaluation, inference, and API deployment.

<!-- Updated path resolution for GitHub Actions compatibility -->

## Project Structure
```
├── data/
│   └── data.csv              # 50-line house dataset
├── src/
│   ├── train.py              # training logic
│   ├── eval.py               # evaluation logic
│   ├── infer.py              # load model + run prediction
│   └── model.py              # model definition/helper
├── api/
│   └── app.py                # FastAPI app for serving predictions
├── artifacts/                # saved models, metrics (created during training)
├── tests/
│   ├── test_data.py          # data validation tests
│   └── test_model.py         # model sanity tests
├── configs/
│   └── train.yaml            # training parameters
├── .github/workflows/
│   └── ci.yml                # GitHub Actions pipeline
├── Dockerfile                # container environment setup
├── docker-compose.yml        # orchestration
├── requirements.txt          # Python dependencies
└── README.md
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model
```bash
python src/train.py
```

### 3. Evaluate Model
```bash
python src/eval.py
```

### 4. Test Inference
```bash
python src/infer.py
```

### 5. Run API Server
```bash
uvicorn api.app:app --reload
```

Then visit http://localhost:8000/docs for the API documentation.

### 6. Run Tests
```bash
pytest tests/ -v
```

## Docker Usage

### Build and Run
```bash
docker-compose up --build
```

### API Endpoints
- `POST /predict` - Make predictions
- `GET /health` - Check API health and model status

### Example API Usage
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "area_sqft": 1200,
       "bedrooms": 3,
       "bathrooms": 2,
       "age_years": 10
     }'
```

## Model Details
- **Algorithm**: Linear Regression
- **Features**: area_sqft, bedrooms, bathrooms, age_years
- **Target**: price
- **Train/Test Split**: 80/20
- **Evaluation Metric**: RMSE
