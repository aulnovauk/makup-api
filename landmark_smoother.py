"""
landmark_smoother.py — One-Euro Filter for MediaPipe Face Landmarks
════════════════════════════════════════════════════════════════════
Gap Fix #1: Temporal landmark jitter elimination.

WHY THIS IS THE HIGHEST-ROI FIX:
  MediaPipe FaceLandmarker emits landmark positions that jump 2–5px
  per frame even when the face is perfectly still. This is measurement
  noise inherent to the neural network's stochastic inference. Every
  makeup edge shimmers at this frequency — lipstick, eyeshadow, and
  blush borders all flicker visibly even at 30fps.

  Banuba uses Kalman-filter temporal smoothing + head-pose prediction
  to produce rock-steady landmarks. The One-Euro filter achieves
  equivalent visual quality with far less implementation complexity:
  two exponential smoothers, zero matrix operations, zero allocations
  per frame after initialisation.

ONE-EURO FILTER PROPERTIES (Casiez et al., CHI 2012):
  • Low jitter at low speed (the user's face is mostly still)
  • Low lag at high speed (when the user turns or moves quickly)
  • Only two parameters: min_cutoff (jitter control) and beta (lag control)
  • Tuned values for face tracking:
      min_cutoff = 1.0  → smooth out ~1-2px noise at rest
      beta       = 0.05 → allow fast response on rapid head turns
      d_cutoff   = 1.0  → standard derivative filter

HOW TO USE:
  smoother = LandmarkSmoother(n_landmarks=478)
  
  # In render loop, after MediaPipe detection:
  t = time.perf_counter()
  smooth_lm = smoother.smooth(t, results.face_landmarks[0])
  
  # smooth_lm is a list of SimpleNamespace with .x, .y, .z, .visibility
  # Drop-in replacement for results.face_landmarks[i]
"""

from __future__ import annotations

import math
from types import SimpleNamespace


# ─────────────────────────────────────────────────────────
#  CORE FILTER  (scalar, per-coordinate)
# ─────────────────────────────────────────────────────────

def _smoothing_factor(t_e: float, cutoff: float) -> float:
    """Compute EMA alpha from elapsed time and cutoff frequency."""
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


def _exp_smooth(alpha: float, x: float, x_prev: float) -> float:
    return alpha * x + (1.0 - alpha) * x_prev


class _ScalarFilter:
    """One-Euro filter for a single scalar signal."""

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "x_prev", "dx_prev", "t_prev", "_init")

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self.x_prev  = 0.0
        self.dx_prev = 0.0
        self.t_prev  = 0.0
        self._init   = False

    def __call__(self, t: float, x: float) -> float:
        if not self._init:
            self.x_prev = x
            self.t_prev = t
            self._init  = True
            return x

        t_e = max(t - self.t_prev, 1e-6)   # guard against zero dt

        # Filtered derivative
        a_d    = _smoothing_factor(t_e, self.d_cutoff)
        dx     = (x - self.x_prev) / t_e
        dx_hat = _exp_smooth(a_d, dx, self.dx_prev)

        # Adaptive cutoff — higher speed → higher cutoff → less smoothing → less lag
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a      = _smoothing_factor(t_e, cutoff)
        x_hat  = _exp_smooth(a, x, self.x_prev)

        self.x_prev  = x_hat
        self.dx_prev = dx_hat
        self.t_prev  = t
        return x_hat

    def reset(self) -> None:
        self._init = False


# ─────────────────────────────────────────────────────────
#  LANDMARK SMOOTHER  (all 478 points, xyz + visibility)
# ─────────────────────────────────────────────────────────

class LandmarkSmoother:
    """
    Applies One-Euro filtering to all landmarks of one face.

    For N landmarks we maintain 3 filters per landmark (x, y, z).
    Visibility is also filtered to prevent abrupt mask toggles.

    Parameters
    ----------
    n_landmarks : int
        Number of face landmarks (MediaPipe 478-point model).
    min_cutoff : float
        Lower = smoother at rest but more lag. 1.0 is good for makeup.
        Try 0.5 if you still see jitter, or 2.0 if motion is too laggy.
    beta : float
        Higher = faster response on fast motion but more jitter.
        0.05 is calibrated for slow-moving face makeup applications.
    """

    def __init__(
        self,
        n_landmarks: int = 478,
        min_cutoff:  float = 1.0,
        beta:        float = 0.05,
        d_cutoff:    float = 1.0,
    ) -> None:
        self._n = n_landmarks
        kw = dict(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        # Three filters per landmark (x, y, z) + visibility
        self._fx  = [_ScalarFilter(**kw) for _ in range(n_landmarks)]
        self._fy  = [_ScalarFilter(**kw) for _ in range(n_landmarks)]
        self._fz  = [_ScalarFilter(**kw) for _ in range(n_landmarks)]
        self._fv  = [_ScalarFilter(min_cutoff=0.5, beta=0.0) for _ in range(n_landmarks)]
        self._last_t = 0.0

    def smooth(
        self,
        t:          float,
        landmarks:  list,   # MediaPipe NormalizedLandmarkList or similar
    ) -> list[SimpleNamespace]:
        """
        Filter a landmark list and return smoothed versions.

        Args:
            t:         Current timestamp (time.perf_counter() is fine)
            landmarks: MediaPipe face landmark list for ONE face

        Returns:
            List of SimpleNamespace objects with .x .y .z .visibility
            Drop-in replacement for MediaPipe landmark objects in _to_pts()
        """
        result = []
        for i, lm in enumerate(landmarks):
            sx = self._fx[i](t, float(lm.x))
            sy = self._fy[i](t, float(lm.y))
            sz = self._fz[i](t, float(getattr(lm, 'z', 0.0)))
            vis = lm.visibility if lm.visibility is not None else 1.0
            sv = self._fv[i](t, float(vis))
            ns = SimpleNamespace(x=sx, y=sy, z=sz, visibility=sv)
            result.append(ns)
        self._last_t = t
        return result

    def reset(self) -> None:
        """Reset all filters (call when face is lost and re-detected)."""
        for f in self._fx + self._fy + self._fz + self._fv:
            f.reset()


# ─────────────────────────────────────────────────────────
#  MULTI-FACE SMOOTHER  (manages one smoother per face)
# ─────────────────────────────────────────────────────────

class MultiFaceSmoother:
    """
    Manages landmark smoothers for up to max_faces faces.

    Handles face appearance/disappearance gracefully:
    - New face detected → fresh smoother (no stale state)
    - Face disappears → smoother released
    - Face reappears → new smoother (clean start)
    """

    def __init__(
        self,
        max_faces:   int   = 2,
        n_landmarks: int   = 478,
        min_cutoff:  float = 1.0,
        beta:        float = 0.05,
    ) -> None:
        self._max  = max_faces
        self._n    = n_landmarks
        self._kw   = dict(n_landmarks=n_landmarks, min_cutoff=min_cutoff, beta=beta)
        self._prev_count = 0
        self._smoothers: list[LandmarkSmoother] = []

    def smooth_all(
        self,
        t:              float,
        face_landmarks: list,   # list of landmark lists, one per face
    ) -> list[list[SimpleNamespace]]:
        """
        Smooth all detected faces. Returns list of smoothed landmark lists.

        If the number of detected faces changes, smoothers are
        created/destroyed to match. This prevents stale filter state
        from a previous face being applied to a newly detected one.
        """
        n = len(face_landmarks)

        # Reset if face count changed — avoids filter state contamination
        if n != self._prev_count:
            self._smoothers = [LandmarkSmoother(**self._kw) for _ in range(n)]
            self._prev_count = n

        # Ensure we have enough smoothers (shouldn't happen but guard anyway)
        while len(self._smoothers) < n:
            self._smoothers.append(LandmarkSmoother(**self._kw))

        return [
            self._smoothers[i].smooth(t, face_landmarks[i])
            for i in range(n)
        ]

    def reset(self) -> None:
        'Reset all smoothers and state. Call when all faces are lost.'
        for s in self._smoothers:
            s.reset()
        self._smoothers = []
        self._prev_count = 0

