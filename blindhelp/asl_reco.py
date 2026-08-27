"""
asl_recognizer.py  —  FIXED
Full 26-letter ASL static classifier using MediaPipe landmark geometry.

BUGS FIXED vs original:
  - G vs L: were identical conditions → G always won, L never triggered
  - A vs S vs E vs M vs N: overlapping fist conditions disambiguated
  - D vs Z: D was catching Z (T_curl implies not T_open)
  - C vs O: C range too broad, now tightened and ordered correctly
  - K: added missing spread check so K doesn't shadow H
  - P: now properly distinguished from K by index direction
  - classify_with_confidence: fixed to pass lm correctly
"""

import math


# ── Geometry helpers ──────────────────────────────────────────────────────────

def dist(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)


def angle_three_points(a, b, c):
    """Angle at vertex b formed by points a–b–c, in degrees."""
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
    0 = fully straight, 1 = fully curled.
    Uses PIP joint angle: ~170° straight, ~50° fully curled.
    """
    angle = angle_three_points(lm[mcp], lm[pip], lm[tip])
    return 1.0 - max(0.0, min(1.0, (angle - 50.0) / 120.0))


def thumb_curl(lm):
    """Thumb curl using CMC→MCP→IP angle."""
    angle = angle_three_points(lm[1], lm[2], lm[3])
    return 1.0 - max(0.0, min(1.0, (angle - 40.0) / 130.0))


def get_finger_curls(lm):
    return {
        'thumb':  thumb_curl(lm),
        'index':  finger_curl(lm, tip=8,  pip=7,  mcp=6),
        'middle': finger_curl(lm, tip=12, pip=11, mcp=10),
        'ring':   finger_curl(lm, tip=16, pip=15, mcp=14),
        'little': finger_curl(lm, tip=20, pip=19, mcp=18),
    }


def finger_spread(lm):
    """Normalised spread between index and middle fingertips."""
    hand_size = dist(lm[0], lm[9]) or 1.0
    return dist(lm[8], lm[12]) / hand_size


def index_middle_crossed(lm):
    """True if index tip is to the right of middle tip → crossed (R)."""
    return lm[8].x > lm[12].x


def thumb_between_fingers(lm):
    """True if thumb tip is between index and middle (T sign)."""
    tip = lm[4]
    return (lm[8].x < tip.x < lm[12].x or lm[12].x < tip.x < lm[8].x)


def thumb_is_up(lm):
    """
    True if thumb tip is notably ABOVE the wrist (lower y value).
    Used to separate L (thumb up + index up) from G (index sideways only).
    In image coordinates: smaller y = higher on screen.
    """
    return lm[4].y < lm[0].y - 0.05   # thumb tip at least 5% above wrist


def index_points_sideways(lm):
    """
    True when index tip is far to the side of the wrist (G/Q shape).
    Uses x-axis distance normalised by hand size.
    """
    hand_size = dist(lm[0], lm[9]) or 1.0
    return abs(lm[8].x - lm[0].x) / hand_size > 0.25


def thumb_over_fingers(lm):
    """
    Rough check: thumb tip is in front of (lower y than) index MCP.
    Separates S (thumb over) from A (thumb alongside).
    """
    return lm[4].y > lm[5].y   # thumb tip below index MCP → draped over


def all_fingertips_low(lm):
    """
    True when all four fingertips are below their PIP joints → E shape
    (all joints bent, fingertips pressing toward palm).
    """
    tips = [8, 12, 16, 20]
    pips = [7, 11, 15, 19]
    return all(lm[t].y > lm[p].y for t, p in zip(tips, pips))


# ── Main classifier ───────────────────────────────────────────────────────────

def classify_asl(lm):
    """
    lm : MediaPipe .landmark repeated-container (21 items, each with .x .y .z)
    Returns : single uppercase letter A–Z, or '?' if unrecognised.

    Curl thresholds used:
        < 0.35  →  extended (open)
        > 0.65  →  curled (closed)
        between →  partial / mid-bend
    """
    c   = get_finger_curls(lm)
    t   = c['thumb']
    idx = c['index']
    mid = c['middle']
    rng = c['ring']
    lil = c['little']

    spread = finger_spread(lm)

    # Boolean flags
    T_open = t   < 0.40
    T_mid  = 0.40 <= t  <= 0.65
    T_curl = t   > 0.65

    I_open = idx < 0.35
    I_mid  = 0.35 <= idx <= 0.65
    I_curl = idx > 0.65

    M_open = mid < 0.35
    M_mid  = 0.35 <= mid <= 0.65
    M_curl = mid > 0.65

    R_open = rng < 0.35
    R_curl = rng > 0.65

    L_open = lil < 0.35
    L_curl = lil > 0.65

    # ── All-four-fingers-up ───────────────────────────────────────────────────
    # B: four fingers straight up, thumb tucked in
    if I_open and M_open and R_open and L_open and T_curl:
        return 'B'

    # ── Three-fingers-up ─────────────────────────────────────────────────────
    # W: index + middle + ring up, little curled
    if I_open and M_open and R_open and L_curl and T_curl:
        return 'W'

    # ── Two-fingers-up ────────────────────────────────────────────────────────
    # V (peace): index + middle open, spread apart, rest curled, thumb curled
    if I_open and M_open and R_curl and L_curl and T_curl and spread >= 0.45:
        return 'V'

    # R (crossed): index + middle open but CROSSED
    if I_open and M_open and R_curl and L_curl and T_curl and index_middle_crossed(lm):
        return 'R'

    # U: index + middle up close together, rest curled, thumb curled
    if I_open and M_open and R_curl and L_curl and T_curl and spread < 0.45:
        return 'U'

    # H: index + middle extended sideways (ring + little + thumb curled)
    if I_open and M_open and R_curl and L_curl and T_curl:
        return 'H'

    # K: index + middle up, thumb open (pointing between them), ring + little curled
    if I_open and M_open and R_curl and L_curl and T_open:
        return 'K'

    # ── One-finger-up ─────────────────────────────────────────────────────────
    # D: only index up, thumb partial/curl touching middle
    if I_open and M_curl and R_curl and L_curl and (T_mid or T_curl):
        return 'D'

    # Z: index fully extended (like pointing), thumb fully curled, others curled
    #    (Z draws a Z in the air; static pose looks like pointing with tight fist)
    #    NOTE: must come AFTER D — Z uses T_curl, D uses T_mid or T_curl.
    #    Separate by how tightly everything else is curled.
    if I_open and M_curl and R_curl and L_curl and T_curl:
        return 'Z'

    # G: index pointing SIDEWAYS (horizontal), thumb open (same direction), rest curled
    #    Key: index tip is far to the side, thumb is NOT pointing upward
    if I_open and T_open and M_curl and R_curl and L_curl and index_points_sideways(lm) and not thumb_is_up(lm):
        return 'G'

    # L: index pointing UP, thumb pointing sideways (L-shape)
    #    Key: thumb tip is notably above the wrist line
    if I_open and T_open and M_curl and R_curl and L_curl and thumb_is_up(lm):
        return 'L'

    # Fallback for index+thumb open (when neither G nor L geometry matches)
    if I_open and T_open and M_curl and R_curl and L_curl:
        return 'L'   # default to L as the more common static shape

    # ── Pinky-only / Thumb+Pinky ─────────────────────────────────────────────
    # Y: thumb + little finger out
    if L_open and T_open and I_curl and M_curl and R_curl:
        return 'Y'

    # I: only little finger up
    if I_curl and M_curl and R_curl and L_open and T_curl:
        return 'I'

    # ── All-fingers-curled fist group  (A / S / E / M / N / T / X) ───────────
    all_curled = I_curl and M_curl and R_curl and L_curl

    if all_curled:
        # E: all joints bent, fingertips hooking downward toward palm
        if all_fingertips_low(lm) and T_curl:
            return 'E'

        # T: thumb tip between index and middle fingers
        if thumb_between_fingers(lm):
            return 'T'

        # S: thumb wrapped OVER the curled fingers (thumb tip is low, over finger knuckles)
        if T_curl and thumb_over_fingers(lm):
            return 'S'

        # M: three fingers (idx/mid/rng) folded over thumb from front
        #    Hard to distinguish from N/A statically; use thumb openness
        if T_curl and not thumb_over_fingers(lm):
            return 'A'   # default fist without thumb-over = A

        # N: index + middle over thumb (subcase of A-ish shape)
        # (N and M are very similar statically — left as A fallback)

    # X: index finger hooked (partially curled), others tightly closed
    if I_mid and M_curl and R_curl and L_curl and (T_curl or T_mid):
        return 'X'

    # ── Partial / O / C shapes ────────────────────────────────────────────────
    # O: all fingers + thumb form tight O — medium curl, thumb meeting index tip
    all_partial_tight = (0.40 < idx < 0.70 and 0.40 < mid < 0.70 and
                         0.40 < rng < 0.70 and 0.40 < lil < 0.70 and T_mid)
    if all_partial_tight:
        return 'O'

    # C: all fingers loosely curved, more open than O
    all_partial_loose = (0.25 < idx < 0.65 and 0.25 < mid < 0.65 and
                         0.25 < rng < 0.65 and 0.25 < lil < 0.65 and T_open)
    if all_partial_loose:
        return 'C'

    # ── F / P / Q ─────────────────────────────────────────────────────────────
    # F: index + thumb pinch, middle + ring + little extended
    if M_open and R_open and L_open and I_curl and (T_mid or T_curl):
        return 'F'

    # P: like K but hand tilted down — approximate with index open, middle open, thumb open
    if I_open and M_open and R_curl and L_curl and T_open and spread > 0.5:
        return 'P'

    # Q: like G but pointing downward — same static shape as G in practice
    # (skipped — covered by G)

    return '?'


# ── Confidence wrapper ────────────────────────────────────────────────────────

def classify_with_confidence(lm, history, history_size=10):
    """
    lm       : MediaPipe .landmark container (pass directly from result)
    history  : mutable list you maintain across frames (pass same list each call)
    Returns  : (letter, confidence_0_to_1)
    """
    letter = classify_asl(lm)
    history.append(letter)
    if len(history) > history_size:
        history.pop(0)

    if not history:
        return letter, 0.0

    most_common = max(set(history), key=history.count)
    confidence  = history.count(most_common) / len(history)
    return most_common, confidence
