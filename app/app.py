import os
import json
import joblib
import numpy as np
from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Optional Redis usage (safe handling)
USE_REDIS = False  # Change to True if you want to enable Redis caching
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
try:
    if USE_REDIS:
        import redis
        r = redis.from_url(REDIS_URL, decode_responses=True)
    else:
        r = None
except ImportError:
    r = None
    print("⚠ Redis not installed — continuing without Redis support.")

# Paths to models
BASE_DIR = os.path.join(os.path.dirname(__file__), "models")
AE_PATH = os.path.join(BASE_DIR, "autoencoder_model.h5")
BILSTM_PATH = os.path.join(BASE_DIR, "bilstm_model.h5")
LGB_PATH = os.path.join(BASE_DIR, "hybrid_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_JSON = os.path.join(BASE_DIR, "feature_columns.json")

# ---------------------- LOAD MODELS ----------------------
import tensorflow as tf
print("🔄 Loading models...")

# Load Keras models with compatibility mode
ae = tf.keras.models.load_model(AE_PATH, compile=False)
bilstm = tf.keras.models.load_model(BILSTM_PATH, compile=False)
lgb = joblib.load(LGB_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURES_JSON, "r") as f:
    FEATURE_COLS = json.load(f)

print(f"✅ Loaded models successfully — {len(FEATURE_COLS)} features.")

# ---------------------- FASTAPI SETUP ----------------------
app = FastAPI(title="Real-Time Credit Card Fraud Detection")

# Allow all origins for local testing (tighten in production)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- HELPERS ----------------------
def make_feature_vector(features_dict: Dict[str, float]) -> np.ndarray:
    """Convert input dict into an ordered feature vector"""
    vec = np.array([features_dict.get(col, 0.0) for col in FEATURE_COLS], dtype=np.float32).reshape(1, -1)
    return vec

def compute_ae_score(x_scaled: np.ndarray) -> float:
    """Compute Autoencoder reconstruction error"""
    recon = ae.predict(x_scaled, verbose=0)
    return float(np.mean(np.square(x_scaled - recon), axis=1)[0])

def compute_seq_score(history_list: List[Dict[str, float]], current_vector_scaled: np.ndarray, window: int = 8) -> float:
    """Compute BiLSTM sequence prediction score"""
    seq = []
    if history_list:
        hist = history_list[-(window - 1):]  # last (window-1) records
        for rec in hist:
            xv = make_feature_vector(rec)
            xv_scaled = scaler.transform(xv)
            seq.append(xv_scaled.flatten())

    seq.append(current_vector_scaled.flatten())

    if len(seq) < window:
        return 0.0

    seq_arr = np.stack(seq[-window:], axis=0).reshape(1, window, -1).astype(np.float32)
    try:
        prob = float(bilstm.predict(seq_arr, verbose=0).flatten()[0])
    except Exception:
        prob = 0.0
    return prob

def push_history(card_id: str, feature_dict: Dict[str, float], maxlen: int = 200):
    """Store transaction history in Redis"""
    if r:
        r.lpush(card_id, json.dumps(feature_dict))
        r.ltrim(card_id, 0, maxlen - 1)

def get_history(card_id: str, window: int):
    """Retrieve transaction history from Redis"""
    if not r:
        return []
    arr = r.lrange(card_id, 0, window - 2)
    return [json.loads(x) for x in arr][::-1]  # oldest → newest

# ---------------------- REST API ----------------------
@app.post("/predict")
def predict_rest(payload: Dict):
    """REST endpoint for fraud prediction"""
    try:
        features = payload.get("features", {})
        history = payload.get("history", None)
        card_id = payload.get("card_id", None)

        x_raw = make_feature_vector(features)
        x_scaled = scaler.transform(x_raw)
        ae_score = compute_ae_score(x_scaled)

        seq_score = 0.0
        if history:
            seq_score = compute_seq_score(history, x_scaled)
        elif card_id and r:
            hist = get_history(card_id, window=8)
            seq_score = compute_seq_score(hist, x_scaled)

        fused = np.concatenate(
            [x_scaled, np.array([[ae_score]]), np.array([[seq_score]])], axis=1
        )

        prob = float(lgb.predict_proba(fused)[:, 1][0])
        is_fraud = int(prob >= 0.5)

        if card_id and r:
            push_history(card_id, features)

        return {
            "fraud_probability": round(prob, 6),
            "is_fraud": is_fraud,
            "ae_score": round(ae_score, 6),
            "seq_score": round(seq_score, 6),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------- WEBSOCKET ----------------------
@app.websocket("/ws/fraud")
async def websocket_fraud(websocket: WebSocket):
    """WebSocket endpoint for real-time fraud detection"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                req = json.loads(data)
            except Exception:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                continue

            features = req.get("features", {})
            card_id = req.get("card_id")
            history = req.get("history")

            x_raw = make_feature_vector(features)
            x_scaled = scaler.transform(x_raw)
            ae_score = compute_ae_score(x_scaled)

            seq_score = 0.0
            if history:
                seq_score = compute_seq_score(history, x_scaled)
            elif card_id and r:
                hist = get_history(card_id, window=8)
                seq_score = compute_seq_score(hist, x_scaled)

            fused = np.concatenate(
                [x_scaled, np.array([[ae_score]]), np.array([[seq_score]])], axis=1
            )

            try:
                prob = float(lgb.predict_proba(fused)[:, 1][0])
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"Prediction error: {str(e)}"}))
                continue

            is_fraud = int(prob >= 0.5)

            if card_id and r:
                push_history(card_id, features)

            resp = {
                "fraud_probability": round(prob, 6),
                "is_fraud": is_fraud,
                "ae_score": round(ae_score, 6),
                "seq_score": round(seq_score, 6),
            }
            await websocket.send_text(json.dumps(resp))

    except WebSocketDisconnect:
        print("🔌 WebSocket client disconnected")   # ✅ Added handler
