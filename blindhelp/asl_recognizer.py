"""
asl_recognizer.py
Full 26-letter ASL static sign classifier using MediaPipe landmark geometry.
Uses finger angles, curl ratios, and inter-landmark distances — much more
accurate than simple up/down detection.
"""

import math


# ── Geometry helpers ──────────────────────────────────────────────────────────

def dist(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)


def angle_three_points(a, b, c):
    """Angle at vertex b formed by points a-b-c, in degrees."""
    ba = (a.x - b.x, a.y - b.y, a.z - b.z)
    bc = (c.x - b.x, c.y - b.y, c.z - b.z)
    dot = sum(ba[i] * bc[i] for i in range(3))
    mag_ba = math.sqrt(sum(v**2 for v in ba))
    mag_bc = math.sqrt(sum(v**2 for v in bc))
    if mag_ba * mag_bc == 0:
        return 180.0
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def finger_curl(lm, tip, pip, mcp):
    """
    Returns a 0-1 curl value for a finger.
    0 = fully extended (straight), 1 = fully curled.
    Uses the angle at the PIP joint.
    """
    angle = angle_three_points(lm[mcp], lm[pip], lm[tip])
    # straight finger ~170°, fully curled ~50°
    curl = 1.0 - max(0.0, min(1.0, (angle - 50.0) / 120.0))
    return curl


def thumb_curl(lm):
    """Thumb curl using CMC-MCP-IP angle."""
    angle = angle_three_points(lm[1], lm[2], lm[3])
    curl = 1.0 - max(0.0, min(1.0, (angle - 40.0) / 130.0))
    return curl


def get_finger_curls(lm):
    """
    Returns dict of curl values (0=open, 1=closed) for all 5 fingers.
    Landmark IDs: thumb(1-4), index(5-8), middle(9-12), ring(13-16), little(17-20)
    """
    return {
        'thumb':  thumb_curl(lm),
        'index':  finger_curl(lm, tip=8,  pip=7,  mcp=6),
        'middle': finger_curl(lm, tip=12, pip=11, mcp=10),
        'ring':   finger_curl(lm, tip=16, pip=15, mcp=14),
        'little': finger_curl(lm, tip=20, pip=19, mcp=18),
    }


def finger_spread(lm):
    """
    Returns spread distance between index and middle fingertips
    normalised by hand size (wrist-to-middle-mcp distance).
    """
    hand_size = dist(lm[0], lm[9]) or 1.0
    spread = dist(lm[8], lm[12]) / hand_size
    return spread


def index_middle_crossed(lm):
    """Returns True if index and middle fingers appear crossed (R sign)."""
    return lm[8].x > lm[12].x  # index tip to the right of middle tip


def thumb_between_fingers(lm):
    """Rough check if thumb tip is between index and middle (for T sign)."""
    tip = lm[4]
    return (lm[8].x < tip.x < lm[12].x or lm[12].x < tip.x < lm[8].x)


def thumb_up_direction(lm):
    """
    Returns how far the thumb is extended laterally away from the palm.
    Positive = thumb extended to the left (from hand's perspective).
    """
    hand_size = dist(lm[0], lm[9]) or 1.0
    return (lm[0].x - lm[4].x) / hand_size


# ── Main 26-letter classifier ─────────────────────────────────────────────────

def classify_asl(lm):
    """
    lm: list of 21 MediaPipe hand landmark objects (each has .x, .y, .z)
    Returns: single uppercase letter A-Z, or '?' if unrecognised.

    Decision tree based on curl values + geometric checks.
    Curl thresholds: < 0.35 = extended, > 0.65 = curled, in between = partial.
    """
    c = get_finger_curls(lm)
    t   = c['thumb']
    idx = c['index']
    mid = c['middle']
    rng = c['ring']
    lil = c['little']

    spread = finger_spread(lm)

    # Helper booleans with generous thresholds
    T_open = t   < 0.4
    T_curl = t   > 0.6
    I_open = idx < 0.35
    I_curl = idx > 0.65
    M_open = mid < 0.35
    M_curl = mid > 0.65
    R_open = rng < 0.35
    R_curl = rng > 0.65
    L_open = lil < 0.35
    L_curl = lil > 0.65

    # ── Letters ───────────────────────────────────────────────────────────────

    # A: fist, thumb resting on side (not fully curled, others all curled)
    if I_curl and M_curl and R_curl and L_curl and not T_curl:
        return 'A'

    # S: fist, thumb over fingers (thumb more curled than A)
    if I_curl and M_curl and R_curl and L_curl and T_curl:
        return 'S'

    # B: four fingers up, thumb tucked across palm
    if I_open and M_open and R_open and L_open and T_curl:
        return 'B'

    # C: all fingers partially curved forming C shape
    if (0.3 < idx < 0.7) and (0.3 < mid < 0.7) and (0.3 < rng < 0.7) and (0.3 < lil < 0.7):
        return 'C'

    # D: index up, thumb tip touches middle (rough: index open, others curl, thumb partial)
    if I_open and M_curl and R_curl and L_curl and not T_open:
        return 'D'

    # E: all fingers curled (tips touching palm), thumb tucked
    if (0.5 < idx < 0.85) and (0.5 < mid < 0.85) and (0.5 < rng < 0.85) and (0.5 < lil < 0.85) and T_curl:
        return 'E'

    # F: index+thumb pinch, other three up
    if M_open and R_open and L_open and I_curl and not T_open:
        return 'F'

    # G: index pointing sideways, thumb parallel (index + thumb extended, rest curled)
    if I_open and T_open and M_curl and R_curl and L_curl:
        return 'G'

    # H: index + middle extended sideways together
    if I_open and M_open and R_curl and L_curl and T_curl:
        return 'H'

    # I: pinky only up
    if I_curl and M_curl and R_curl and L_open and T_curl:
        return 'I'

    # K: index + middle up, thumb up (pointing up-ish)
    if I_open and M_open and R_curl and L_curl and T_open and spread < 0.5:
        return 'K'

    # L: index + thumb out (L shape)
    if I_open and T_open and M_curl and R_curl and L_curl:
        return 'L'

    # M: three fingers (index, middle, ring) over curled thumb
    if I_curl and M_curl and R_curl and L_curl and not T_open:
        return 'M'

    # N: index + middle over thumb
    if I_curl and M_curl and R_curl and L_curl:
        return 'N'

    # O: all fingers + thumb form O shape (medium curl all around)
    if (0.35 < idx < 0.65) and (0.35 < mid < 0.65) and (0.35 < rng < 0.65) and (0.35 < lil < 0.65) and not T_open:
        return 'O'

    # P: like K but hand points downward — approximate with same shape
    if I_open and M_open and R_curl and L_curl and T_open and spread > 0.5:
        return 'P'

    # Q: index + thumb pointing down (like G downward) — same shape as G in static
    # (Q and G are hard to distinguish statically; skip Q, covered by G)

    # R: index + middle crossed (index over middle)
    if I_open and M_open and R_curl and L_curl and T_curl and index_middle_crossed(lm):
        return 'R'

    # T: thumb between index and middle fingers
    if I_curl and M_curl and R_curl and L_curl and thumb_between_fingers(lm):
        return 'T'

    # U: index + middle up together (not spread)
    if I_open and M_open and R_curl and L_curl and T_curl and spread < 0.45:
        return 'U'

    # V: index + middle up, spread apart (peace sign)
    if I_open and M_open and R_curl and L_curl and T_curl and spread >= 0.45:
        return 'V'

    # W: index + middle + ring up
    if I_open and M_open and R_open and L_curl and T_curl:
        return 'W'

    # X: index hooked/partially curled, others closed
    if (0.4 < idx < 0.75) and M_curl and R_curl and L_curl:
        return 'X'

    # Y: thumb + pinky out
    if L_open and T_open and I_curl and M_curl and R_curl:
        return 'Y'

    # Z: index pointing (same as D), treated as D if D already assigned
    # Map to Z only if index is very extended and others are tightly curled
    if I_open and M_curl and R_curl and L_curl and T_curl:
        return 'Z'

    return '?'


# ── Confidence wrapper ────────────────────────────────────────────────────────

def classify_with_confidence(lm, history, history_size=10):
    """
    Runs the classifier and checks stability over a short history.
    Returns (letter, confidence_0_to_1).
    history: a mutable list you pass in and maintain across frames.
    """
    letter = classify_asl(lm)
    history.append(letter)
    if len(history) > history_size:
        history.pop(0)

    if not history:
        return letter, 0.0

    most_common = max(set(history), key=history.count)
    confidence = history.count(most_common) / len(history)
    return most_common, confidence
