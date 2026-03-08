"""
SessionManager: wraps IntegratedPsychologistAI for use by FastAPI.

Runs the camera + AI loop in a background thread.
Exposes get_latest_frame() and get_latest_state() for MJPEG/WebSocket endpoints.
"""

import threading
import time
import sys
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Any

import cv2

# Make sure the project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.integrated_psychologist_ai import IntegratedPsychologistAI


class SessionManager:
    """
    Thread-safe wrapper around IntegratedPsychologistAI.

    Usage:
        session_manager.start(user_id)   # begins camera loop in background
        session_manager.stop()           # saves session, stops camera
        state = session_manager.get_latest_state()
        frame = session_manager.get_latest_frame()
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._ai: Optional[IntegratedPsychologistAI] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_state_dict: Optional[Dict[str, Any]] = None
        self._active_user_id: Optional[str] = None
        self._frame_count = 0
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def active_user_id(self) -> Optional[str]:
        return self._active_user_id

    def start(self, user_id: str) -> None:
        """Initialize AI for user_id and start camera loop."""
        if self._running:
            self.stop()

        self._active_user_id = user_id
        self._frame_count = 0
        self._start_time = time.time()

        # Init AI (loads models — may take a few seconds)
        self._ai = IntegratedPsychologistAI(user_id=user_id)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop camera loop and persist session data."""
        self._running = False

        if self._cap:
            self._cap.release()
            self._cap = None

        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        if self._ai:
            # Stop microphone
            try:
                self._ai.audio_thread.stop()
            except Exception:
                pass

            # Save session (mirrors integrated_psychologist_ai.py finally block)
            if self._frame_count > 0:
                try:
                    session_metrics = self._ai.phase4.session_memory.calculate_metrics()
                    has_new_data = self._ai.phase4.session_memory.get_frame_count() > 0

                    if has_new_data:
                        self._ai.phase4.long_term_memory.add_session(session_metrics)

                        from inference.phase4_user_manager import UserManager
                        user_manager = UserManager(storage_dir="data/user_memory")
                        user_manager.increment_session_count(self._active_user_id)

                        # Phase 5 PSV update
                        p5 = self._ai.phase4.phase5_engine
                        if p5 is not None:
                            p5.add_session(session_metrics)
                            if p5.can_infer_personality():
                                recent_dates = sorted(
                                    self._ai.phase4.long_term_memory.daily_profiles.keys()
                                )[-7:]
                                recent_profiles = [
                                    self._ai.phase4.long_term_memory.daily_profiles[d]
                                    for d in recent_dates
                                ]
                                p5.update_psv(recent_profiles)

                        print(f"[SessionManager] Session saved for user {self._active_user_id}")

                except Exception as e:
                    print(f"[SessionManager] Error saving session: {e}")
                    import traceback
                    traceback.print_exc()

            self._ai = None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_latest_state(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._latest_state_dict) if self._latest_state_dict else None

    # ------------------------------------------------------------------ #
    # Background capture loop                                              #
    # ------------------------------------------------------------------ #

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(0)
        self._cap = cap

        if not cap.isOpened():
            print("[SessionManager] ERROR: Cannot open webcam")
            self._running = False
            return

        print("[SessionManager] Camera started")

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            try:
                # Run all four AI phases
                frame = self._ai.process_frame(frame)
                # Draw info overlay
                frame = self._ai.draw_info_panel(frame)

                self._frame_count += 1
                state = self._ai.latest_state
                phase4 = self._ai.latest_phase4_profile

                elapsed = time.time() - self._start_time
                fps = self._frame_count / elapsed if elapsed > 0 else 0.0

                with self._lock:
                    self._latest_frame = frame.copy()
                    if state:
                        self._latest_state_dict = self._state_to_dict(state, phase4, fps)

            except Exception as e:
                print(f"[SessionManager] Frame processing error: {e}")

            time.sleep(0.033)  # cap at ~30 fps

        cap.release()
        self._cap = None
        print("[SessionManager] Camera stopped")

    # ------------------------------------------------------------------ #
    # State serialization                                                   #
    # ------------------------------------------------------------------ #

    def _state_to_dict(self, state, phase4, fps: float) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "face_emotion": state.dominant_emotion,
            "hidden_emotion": state.hidden_emotion,
            "mental_state": state.mental_state.value,
            "confidence": round(state.confidence, 3),
            "risk_level": state.risk_level.value,
            "stability_score": round(state.stability_score, 3),
            "explanations": state.explanations[:3] if state.explanations else [],
            "timestamp": time.time(),
            "fps": round(fps, 1),
        }

        # Raw signal values (voice emotion, stress level)
        if hasattr(state, "raw_signals") and state.raw_signals:
            for key, signal in state.raw_signals.items():
                if hasattr(signal, "value") and hasattr(signal, "confidence"):
                    result[f"{key}_emotion"] = signal.value
                    result[f"{key}_confidence"] = round(float(signal.confidence), 3)

        if phase4:
            result["adjusted_risk"] = phase4.adjusted_risk.value
            result["risk_adjustment_reason"] = phase4.risk_adjustment_reason

            pers = phase4.personality
            if pers and pers.confidence > 0.3:
                result["personality"] = {
                    "emotional_reactivity": round(pers.emotional_reactivity, 3),
                    "stress_tolerance": round(pers.stress_tolerance, 3),
                    "emotional_stability": round(pers.emotional_stability, 3),
                    "baseline_mood": pers.baseline_mood,
                    "confidence": round(pers.confidence, 3),
                    "data_days": pers.data_days,
                }

            if phase4.deviations:
                result["deviations"] = [
                    {
                        "type": d.deviation_type,
                        "severity": round(d.severity, 3),
                        "description": getattr(d, "description", ""),
                    }
                    for d in phase4.deviations[:3]
                ]

        return result


# Singleton used by all routers
session_manager = SessionManager()
