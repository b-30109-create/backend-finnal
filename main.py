from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle, numpy as np, os

app = FastAPI(title="Deepfake Detection API")

# ── CORS: allow all origins (swap for your Vercel URL in production) ──────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(MODEL_PATH, "rb") as f:
    payload = pickle.load(f)

pipeline       = payload["pipeline"]
label_encoders = payload["label_encoders"]
feature_names  = payload["feature_names"]
target_classes = payload["target_classes"]   # {0: "Real", 1: "Fake"}

# ── Request schema ────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    media_type: str
    content_category: str
    face_count: int
    audio_present: int          # 1 = Yes, 0 = No
    lip_sync_score: float
    visual_artifacts_score: float
    compression_level: float
    lighting_inconsistency_score: float
    source_platform: str
    generation_method: str

# ── Encode categorical fields using saved LabelEncoders ───────────────────────
def encode(request: PredictRequest) -> np.ndarray:
    cat_cols = ["media_type", "content_category", "source_platform", "generation_method"]
    row = {}
    for col in cat_cols:
        le  = label_encoders[col]
        val = getattr(request, col)
        row[col] = int(le.transform([val])[0])
    row["face_count"]                    = request.face_count
    row["audio_present"]                 = request.audio_present
    row["lip_sync_score"]                = request.lip_sync_score
    row["visual_artifacts_score"]        = request.visual_artifacts_score
    row["compression_level"]             = request.compression_level
    row["lighting_inconsistency_score"]  = request.lighting_inconsistency_score
    return np.array([[row[f] for f in feature_names]])

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "model": payload["model_name"]}

@app.post("/predict")
def predict(req: PredictRequest):
    X       = encode(req)
    pred    = int(pipeline.predict(X)[0])
    proba   = pipeline.predict_proba(X)[0]
    return {
        "prediction": target_classes[pred],
        "confidence": round(float(max(proba)) * 100, 2),
        "prob_real":  round(float(proba[0]) * 100, 2),
        "prob_fake":  round(float(proba[1]) * 100, 2),
    }

@app.get("/options")
def options():
    """Return all valid dropdown values for the frontend."""
    return {col: list(le.classes_) for col, le in label_encoders.items()}
