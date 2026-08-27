"""
gesture_library.py
Manages the reference gesture library — stores landmark sequences
with their meanings, supports recording live and importing from video files.
Each entry: { "name": str, "meaning": str, "sequence": [[21 x (x,y,z)], ...] }
Persisted as JSON on disk.
"""

import json
import os
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path

LIBRARY_PATH = "gesture_library.json"

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ── Persistence ───────────────────────────────────────────────────────────────

def load_library() -> list:
    if not Path(LIBRARY_PATH).exists():
        return []
    with open(LIBRARY_PATH, "r") as f:
        return json.load(f)


def save_library(lib: list):
    with open(LIBRARY_PATH, "w") as f:
        json.dump(lib, f, indent=2)


def add_entry(lib: list, name: str, meaning: str, sequence: list) -> list:
    """Add or replace an entry by name."""
    lib = [e for e in lib if e["name"].lower() != name.lower()]
    lib.append({"name": name, "meaning": meaning, "sequence": sequence})
    save_library(lib)
    return lib


def delete_entry(lib: list, name: str) -> list:
    lib = [e for e in lib if e["name"].lower() != name.lower()]
    save_library(lib)
    return lib


# ── Landmark extraction ───────────────────────────────────────────────────────

def landmarks_to_list(hand_landmarks) -> list:
    """Convert MediaPipe hand landmarks to a flat list of (x,y,z) tuples."""
    return [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]


def normalise_frame(frame_lms: list) -> np.ndarray:
    """
    Normalise a 21-landmark frame so it's translation and scale invariant.
    Reference point: wrist (landmark 0).
    Scale: distance from wrist to middle MCP (landmark 9).
    Returns numpy array shape (21, 3).
    """
    pts = np.array(frame_lms, dtype=np.float32)
    origin = pts[0].copy()
    pts -= origin
    scale = np.linalg.norm(pts[9]) or 1.0
    pts /= scale
    return pts


def sequence_to_numpy(sequence: list) -> np.ndarray:
    """Convert list of landmark frames to numpy array (N, 21, 3)."""
    return np.array([normalise_frame(f) for f in sequence], dtype=np.float32)


# ── DTW similarity ────────────────────────────────────────────────────────────

def frame_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Euclidean distance between two normalised landmark frames."""
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    Dynamic Time Warping distance between two landmark sequences.
    Allows flexible timing/speed matching.
    Returns a normalised distance (lower = more similar).
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float("inf")

    # DTW matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = frame_distance(seq_a[i - 1], seq_b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j],
                                   dtw[i, j - 1],
                                   dtw[i - 1, j - 1])

    # Normalise by path length
    return dtw[n, m] / (n + m)


def find_best_match(query_seq: list, library: list,
                    threshold: float = 0.18) -> dict | None:
    """
    Compare query_seq against every library entry using DTW.
    Returns the best matching entry dict, or None if no match within threshold.
    threshold: flexible ~0.18, strict ~0.10, fuzzy ~0.28
    """
    if not library or not query_seq:
        return None

    query_np = sequence_to_numpy(query_seq)
    best_dist  = float("inf")
    best_entry = None

    for entry in library:
        ref_np = sequence_to_numpy(entry["sequence"])
        dist   = dtw_distance(query_np, ref_np)
        if dist < best_dist:
            best_dist  = dist
            best_entry = entry

    if best_dist <= threshold:
        return best_entry
    return None


# ── Record from live camera ───────────────────────────────────────────────────

def record_from_camera(duration_seconds: int = 5,
                        countdown: int = 3,
                        frame_callback=None) -> list:
    """
    Opens camera, counts down, then records hand landmark sequences
    for `duration_seconds` seconds.
    frame_callback(frame): called each frame so caller can display it.
    Returns list of landmark frames (each a list of 21 [x,y,z]).
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sequence   = []
    phase      = "countdown"  # countdown → recording → done
    start_time = None
    cd_start   = None

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75
    ) as hands:

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now   = cv2.getTickCount() / cv2.getTickFrequency()

            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    result.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 220, 120), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=2)
                )

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 90), (10, 14, 20), -1)

            if phase == "countdown":
                if cd_start is None:
                    cd_start = now
                elapsed = now - cd_start
                remaining = int(countdown - elapsed) + 1
                if elapsed >= countdown:
                    phase      = "recording"
                    start_time = now
                else:
                    cv2.putText(frame, f"Get ready... {remaining}",
                                (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                1.8, (0, 210, 255), 3)

            elif phase == "recording":
                elapsed = now - start_time
                remaining = max(0.0, duration_seconds - elapsed)

                # Record landmarks
                if result.multi_hand_landmarks:
                    lm_list = landmarks_to_list(result.multi_hand_landmarks[0])
                    sequence.append(lm_list)

                # Progress bar
                prog = int((elapsed / duration_seconds) * (w - 40))
                cv2.rectangle(frame, (20, 65), (20 + prog, 82), (0, 255, 120), -1)
                cv2.putText(frame, f"● RECORDING  {remaining:.1f}s",
                            (30, 55), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 255, 100), 3)

                if elapsed >= duration_seconds:
                    phase = "done"

            elif phase == "done":
                cv2.putText(frame, f"Done! {len(sequence)} frames captured.",
                            (30, 55), cv2.FONT_HERSHEY_SIMPLEX,
                            1.4, (0, 255, 255), 3)
                if frame_callback:
                    frame_callback(frame)
                cv2.imshow("Recording — press any key to finish", frame)
                cv2.waitKey(1500)
                break

            if frame_callback:
                frame_callback(frame)

            cv2.imshow("Recording — press Q to cancel", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                sequence = []
                break

    cap.release()
    cv2.destroyAllWindows()
    return sequence


# ── Extract from video file ───────────────────────────────────────────────────

def extract_from_video(video_path: str,
                        progress_callback=None) -> list:
    """
    Extract hand landmark sequence from a video file.
    progress_callback(fraction 0-1): optional progress reporting.
    Returns list of landmark frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sequence     = []
    frame_idx    = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.70,
        min_tracking_confidence=0.70
    ) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            if result.multi_hand_landmarks:
                lm_list = landmarks_to_list(result.multi_hand_landmarks[0])
                sequence.append(lm_list)
            frame_idx += 1
            if progress_callback and total_frames > 0:
                progress_callback(frame_idx / total_frames)

    cap.release()
    return sequence
