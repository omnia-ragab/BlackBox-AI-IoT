"""
BLACKBOX — Sensor Simulator + Live Streaming Monitor
=====================================================
Simulates one machine's vibration / temperature / acoustic sensor over
several "weeks" (compressed to seconds of real runtime), passing through
the exact phases described in the BlackBox docs (section 5):

    Week 1        -> baseline learning (HEALTHY, model calibrates)
    Week 2        -> normal operation (HEALTHY)
    Week 3-4      -> early degradation (WATCH)
    Week 4-5      -> advanced degradation (WARNING / CRITICAL)
    Post-maintenance -> back to HEALTHY, new baseline

Each simulated minute produces ONE sensor reading (mirrors the real device
reporting a rolling summary every 60 seconds), which is fed to the model
one at a time — exactly how it would run in production, not as a batch.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

RNG = np.random.default_rng(7)

# ---------------------------------------------------------------------------
# 1) MACHINE SIMULATOR — generates one sensor reading per simulated minute
# ---------------------------------------------------------------------------
class MachineSimulator:
    """
    Produces (vibration_mm_s, temperature_c, acoustic_db) readings for a
    single machine across a full degradation-to-maintenance cycle.
    """

    def __init__(self):
        # baseline (healthy) operating point — mirrors the doc's example numbers
        self.base_vibration = 5.0     # mm/s
        self.base_temp = 60.0         # deg C
        self.base_acoustic = 55.0     # dB

    def generate_run(self, n_baseline=10_080, n_healthy=10_080,
                      n_watch=20_160, n_warning=10_080, n_post=5_040):
        """
        Minute-by-minute reading counts (defaults ~ 1 week / 2 weeks / etc.
        at 1 reading/minute). Reduced for demo speed by the caller.
        """
        readings = []
        phase_labels = []

        def emit(vib, temp, ac, phase):
            readings.append((vib, temp, ac))
            phase_labels.append(phase)

        # --- Phase 1: baseline learning (healthy, just noisy) ---
        for _ in range(n_baseline):
            emit(
                self.base_vibration + RNG.normal(0, 0.15),
                self.base_temp + RNG.normal(0, 0.5),
                self.base_acoustic + RNG.normal(0, 1.0),
                "HEALTHY (baseline)"
            )

        # --- Phase 2: normal operation ---
        for _ in range(n_healthy):
            emit(
                self.base_vibration + RNG.normal(0, 0.15),
                self.base_temp + RNG.normal(0, 0.5),
                self.base_acoustic + RNG.normal(0, 1.0),
                "HEALTHY"
            )

        # --- Phase 3: early -> advanced degradation (gradual ramp) ---
        for i in range(n_watch):
            t = i / n_watch  # 0 -> 1
            drift = t ** 1.5  # slow start, accelerates — like real bearing wear
            emit(
                self.base_vibration + 1.0 * drift + RNG.normal(0, 0.2),
                self.base_temp + 2.0 * drift + RNG.normal(0, 0.5),
                self.base_acoustic + 3.0 * drift + RNG.normal(0, 1.0),
                "WATCH" if drift < 0.5 else "WARNING (early)"
            )

        # --- Phase 4: advanced degradation, close to failure ---
        for i in range(n_warning):
            t = i / n_warning
            drift = 1.0 + 6.0 * (t ** 2)  # steep acceleration toward failure
            emit(
                self.base_vibration + 1.0 + 6.0 * (t ** 2) + RNG.normal(0, 0.3),
                self.base_temp + 2.0 + 13.0 * (t ** 2) + RNG.normal(0, 0.6),
                self.base_acoustic + 3.0 + 12.0 * (t ** 2) + RNG.normal(0, 1.2),
                "WARNING" if t < 0.7 else "CRITICAL"
            )

        # --- Phase 5: post-maintenance, back to healthy ---
        for _ in range(n_post):
            emit(
                self.base_vibration + RNG.normal(0, 0.15),
                self.base_temp + RNG.normal(0, 0.5),
                self.base_acoustic + RNG.normal(0, 1.0),
                "HEALTHY (post-maintenance)"
            )

        df = pd.DataFrame(readings, columns=["vibration", "temperature", "acoustic"])
        df["minute"] = range(len(df))
        df["true_phase"] = phase_labels
        return df


# ---------------------------------------------------------------------------
# 2) LIVE MONITOR — trains on the baseline, then scores readings ONE AT A TIME
# ---------------------------------------------------------------------------
class LiveMonitor:
    """
    SMOOTH_WINDOW: raw per-minute scores are noisy by nature (one sensor
    blip shouldn't flip the status) — we average the last N raw scores,
    matching the doc's idea of the dashboard reporting a rolling trend,
    not an instantaneous spike.

    PERSISTENCE: a status change only takes effect once the new zone has
    held for PERSISTENCE consecutive readings. This is standard alerting
    practice (avoids "flapping" alarms) and mirrors the WATCH/WARNING
    bands in the doc, which describe sustained conditions, not blips.
    """

    def __init__(self, smooth_window=15, persistence=10):
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        self.smooth_window = smooth_window
        self.persistence = persistence
        self.raw_history = []
        self.confirmed_status = "HEALTHY"
        self.candidate_status = "HEALTHY"
        self.candidate_count = 0
        # fixed reference range, set once at calibration time — NOT
        # recomputed per-reading, so a single extreme point can't
        # rescale everything that came before it
        self.calib_lo = None
        self.calib_hi = None

    def calibrate(self, baseline_readings):
        """Phase 1 of real deployment: 7-day baseline learning."""
        X = self.scaler.fit_transform(baseline_readings[["vibration", "temperature", "acoustic"]])
        self.model.fit(X)
        calib_scores = self.model.decision_function(X)
        # reference band = healthy-period score spread, padded a bit so
        # normal noise doesn't already read as "critical" on day one
        spread = calib_scores.max() - calib_scores.min()
        self.calib_hi = calib_scores.max() + 0.25 * spread
        self.calib_lo = calib_scores.min() - 2.5 * spread  # degraded scores go well below this

    @staticmethod
    def _status_from_score(score):
        return (
            "HEALTHY" if score < 0.30 else
            "WATCH" if score < 0.55 else
            "WARNING" if score < 0.80 else
            "CRITICAL"
        )

    def score_one(self, reading_row):
        """
        Score a SINGLE incoming reading — this is the production code path:
        one row in, one status out, no look-ahead. Returns the smoothed
        score and the CONFIRMED status (post-persistence-filter), plus the
        raw instantaneous status for transparency.
        """
        X = self.scaler.transform([[reading_row["vibration"],
                                     reading_row["temperature"],
                                     reading_row["acoustic"]]])
        raw = self.model.decision_function(X)[0]
        self.raw_history.append(raw)

        # smooth: average of the last N raw scores (fixed reference band)
        window = self.raw_history[-self.smooth_window:]
        avg_raw = sum(window) / len(window)
        score = (self.calib_hi - avg_raw) / (self.calib_hi - self.calib_lo)
        score = min(max(score, 0.0), 1.0)  # clip to [0, 1]

        instantaneous_status = self._status_from_score(score)

        # persistence filter: only "confirm" a status change after it
        # holds for `persistence` consecutive readings
        if instantaneous_status == self.candidate_status:
            self.candidate_count += 1
        else:
            self.candidate_status = instantaneous_status
            self.candidate_count = 1

        if self.candidate_count >= self.persistence:
            self.confirmed_status = self.candidate_status

        return score, self.confirmed_status


# ---------------------------------------------------------------------------
# 3) RUN THE FULL SIMULATION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Scaled-down counts for a fast demo run (real deployment: minutes over weeks)
    sim = MachineSimulator()
    run_df = sim.generate_run(
        n_baseline=300, n_healthy=200, n_watch=400, n_warning=200, n_post=150
    )

    monitor = LiveMonitor()
    monitor.calibrate(run_df.iloc[:300])  # first 300 minutes = baseline phase

    scores, statuses = [], []
    last_status = None
    alerts = []

    for _, row in run_df.iterrows():
        score, status = monitor.score_one(row)
        scores.append(score)
        statuses.append(status)
        if status != last_status:
            alerts.append((row["minute"], last_status, status))
            last_status = status

    run_df["anomaly_score"] = scores
    run_df["predicted_status"] = statuses

    print("=== Status transitions detected live (no look-ahead) ===")
    for minute, old, new in alerts:
        print(f"  minute {minute:>4}: {old or 'START'} -> {new}")

    # --- plot: predicted status stream vs. true simulated phase ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(run_df["minute"], run_df["vibration"], label="vibration", alpha=0.7)
    axes[0].plot(run_df["minute"], run_df["temperature"], label="temperature", alpha=0.7)
    axes[0].plot(run_df["minute"], run_df["acoustic"], label="acoustic", alpha=0.7)
    axes[0].set_ylabel("Raw sensor reading")
    axes[0].set_title("BlackBox — Simulated Raw Sensor Stream")
    axes[0].legend(loc="upper left")

    axes[1].plot(run_df["minute"], run_df["anomaly_score"], color="#2563eb")
    axes[1].axhline(0.30, color="green", linestyle="--", alpha=0.5)
    axes[1].axhline(0.55, color="orange", linestyle="--", alpha=0.5)
    axes[1].axhline(0.80, color="red", linestyle="--", alpha=0.5)
    for minute, old, new in alerts:
        axes[1].axvline(minute, color="black", alpha=0.15)
    axes[1].set_ylabel("Live anomaly score")
    axes[1].set_xlabel("Simulated minute")
    axes[1].set_title("BlackBox — Live Model Output (streamed one reading at a time)")

    plt.tight_layout()
    plt.savefig("live_simulation.png", dpi=120)
    plt.close()

    run_df.to_csv("simulated_run_scored.csv", index=False)
    print("\nSaved: live_simulation.png, simulated_run_scored.csv")
