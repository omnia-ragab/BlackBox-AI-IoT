"""
BLACKBOX — API (FastAPI)
=========================
Wraps the same LiveMonitor logic already tested in sensor_simulator.py
behind three HTTP endpoints, one machine at a time:

    POST /calibrate/{machine_id}   -> train on baseline readings
    POST /score/{machine_id}       -> score ONE live reading
    GET  /status/{machine_id}      -> latest known status

Run locally:
    pip install fastapi "uvicorn[standard]"
    uvicorn app:app --reload --port 8000

Then open http://localhost:8000/docs for an interactive test UI (Swagger),
or use the test_api.py script included alongside this file.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="BlackBox Predictive Maintenance API", version="1.0")


# ---------------------------------------------------------------------------
# Data models (request/response shapes — FastAPI validates these automatically)
# ---------------------------------------------------------------------------
class SensorReading(BaseModel):
    vibration: float
    temperature: float
    acoustic: float


class CalibrateRequest(BaseModel):
    baseline_readings: List[SensorReading]


class ScoreResponse(BaseModel):
    machine_id: str
    anomaly_score: float
    status: str
    minute: int


# ---------------------------------------------------------------------------
# LiveMonitor — identical logic to sensor_simulator.py (kept in sync)
# ---------------------------------------------------------------------------
class LiveMonitor:
    def __init__(self, smooth_window=15, persistence=10):
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        self.smooth_window = smooth_window
        self.persistence = persistence
        self.raw_history = []
        self.confirmed_status = "HEALTHY"
        self.candidate_status = "HEALTHY"
        self.candidate_count = 0
        self.calib_lo = None
        self.calib_hi = None
        self.minute = 0
        self.is_calibrated = False

    def calibrate(self, baseline_readings: List[SensorReading]):
        rows = [[r.vibration, r.temperature, r.acoustic] for r in baseline_readings]
        X = self.scaler.fit_transform(rows)
        self.model.fit(X)
        calib_scores = self.model.decision_function(X)
        spread = calib_scores.max() - calib_scores.min()
        self.calib_hi = calib_scores.max() + 0.25 * spread
        self.calib_lo = calib_scores.min() - 2.5 * spread
        self.is_calibrated = True

    @staticmethod
    def _status_from_score(score):
        return (
            "HEALTHY" if score < 0.30 else
            "WATCH" if score < 0.55 else
            "WARNING" if score < 0.80 else
            "CRITICAL"
        )

    def score_one(self, reading: SensorReading):
        X = self.scaler.transform([[reading.vibration, reading.temperature, reading.acoustic]])
        raw = self.model.decision_function(X)[0]
        self.raw_history.append(raw)
        self.minute += 1

        window = self.raw_history[-self.smooth_window:]
        avg_raw = sum(window) / len(window)
        score = (self.calib_hi - avg_raw) / (self.calib_hi - self.calib_lo)
        score = min(max(score, 0.0), 1.0)

        instantaneous_status = self._status_from_score(score)
        if instantaneous_status == self.candidate_status:
            self.candidate_count += 1
        else:
            self.candidate_status = instantaneous_status
            self.candidate_count = 1
        if self.candidate_count >= self.persistence:
            self.confirmed_status = self.candidate_status

        return score, self.confirmed_status


# ---------------------------------------------------------------------------
# In-memory store: one LiveMonitor per machine_id
# (Demo only — a real deployment would persist this, e.g. in Redis/Postgres,
# so state survives an API restart. Swapping this dict for a DB-backed
# store is the only change needed to make it production-durable.)
# ---------------------------------------------------------------------------
monitors: Dict[str, LiveMonitor] = {}


def get_monitor(machine_id: str) -> LiveMonitor:
    if machine_id not in monitors:
        raise HTTPException(
            status_code=404,
            detail=f"Machine '{machine_id}' not calibrated yet — "
                   f"call POST /calibrate/{machine_id} first."
        )
    return monitors[machine_id]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/calibrate/{machine_id}")
def calibrate(machine_id: str, req: CalibrateRequest):
    """
    Phase 1 of deployment: train on ~7 days of healthy baseline readings.
    Must be called once per machine before /score can be used.
    """
    if len(req.baseline_readings) < 20:
        raise HTTPException(
            status_code=400,
            detail="Need at least 20 baseline readings for a stable calibration."
        )
    monitor = LiveMonitor()
    monitor.calibrate(req.baseline_readings)
    monitors[machine_id] = monitor
    return {
        "machine_id": machine_id,
        "status": "calibrated",
        "baseline_readings_used": len(req.baseline_readings),
    }


@app.post("/score/{machine_id}", response_model=ScoreResponse)
def score(machine_id: str, reading: SensorReading):
    """
    Production path: ONE reading in, ONE status out. Call this every time
    the physical sensor reports (e.g. once a minute).
    """
    monitor = get_monitor(machine_id)
    anomaly_score, status = monitor.score_one(reading)
    return ScoreResponse(
        machine_id=machine_id,
        anomaly_score=round(anomaly_score, 4),
        status=status,
        minute=monitor.minute,
    )


@app.get("/status/{machine_id}", response_model=ScoreResponse)
def get_status(machine_id: str):
    """Latest known status without submitting a new reading."""
    monitor = get_monitor(machine_id)
    last_score = monitor.raw_history[-1] if monitor.raw_history else 0.0
    return ScoreResponse(
        machine_id=machine_id,
        anomaly_score=round(last_score, 4),
        status=monitor.confirmed_status,
        minute=monitor.minute,
    )


@app.get("/")
def root():
    return {"service": "BlackBox Predictive Maintenance API", "status": "running"}
