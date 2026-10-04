"""
Height above the table from LiDAR depth (Record3D).  Owner: Dev 2.

The page lies flat on the table, so paper, print, ink and hand SHADOWS all have height ~0, while
the hand and the pen are 1-10 cm up. The table is fitted as a plane in inverse depth: for a
pinhole camera 1/z is exactly linear in the pixel coordinates (u, v) for any plane, so no camera
intrinsics are needed. The fit is robust (RANSAC) and is refreshed every few frames using only
pixels that were on the table, so a hand in view doesn't tilt it.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

RAISED_M = 0.012       # higher than this above the table = hand / pen, not paper


def fit_inverse_plane(depth: np.ndarray, mask: np.ndarray, iters: int = 60, tol_m: float = 0.006,
                      rng: Optional[np.random.Generator] = None) -> Optional[Tuple[float, float, float]]:
    """(a, b, c) with 1/z = a*u + b*v + c over the dominant plane among mask pixels."""
    rng = rng or np.random.default_rng(0)
    vs, us = np.nonzero(mask)
    if len(us) < 50:
        return None
    if len(us) > 1500:
        pick = rng.choice(len(us), 1500, replace=False)
        us, vs = us[pick], vs[pick]
    z = depth[vs, us].astype(np.float64)
    A = np.column_stack([us, vs, np.ones_like(us)]).astype(np.float64)
    inv = 1.0 / z
    best, best_n = None, 0
    for _ in range(iters):
        idx = rng.choice(len(us), 3, replace=False)
        try:
            p = np.linalg.solve(A[idx], inv[idx])
        except np.linalg.LinAlgError:
            continue
        pred = A @ p
        ok = pred > 1e-6
        n = int(np.count_nonzero(ok & (np.abs(1.0 / np.where(ok, pred, 1.0) - z) < tol_m)))
        if n > best_n:
            best, best_n = p, n
    if best is None or best_n < 0.3 * len(us):
        return None
    pred = A @ best
    inl = (pred > 1e-6) & (np.abs(1.0 / np.where(pred > 1e-6, pred, 1.0) - z) < tol_m)
    p, *_ = np.linalg.lstsq(A[inl], inv[inl], rcond=None)
    return float(p[0]), float(p[1]), float(p[2])


class TablePlane:
    """Per-frame height map (metres above the table, NaN where depth is invalid)."""

    def __init__(self, refit_every: int = 15) -> None:
        self.refit_every = refit_every
        self.plane: Optional[Tuple[float, float, float]] = None
        self._n = 0
        self._last_height: Optional[np.ndarray] = None
        self._grid: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def reset(self) -> None:
        self.plane, self._last_height, self._n = None, None, 0

    def height(self, depth: np.ndarray, confidence: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        depth = np.asarray(depth, np.float32)
        valid = np.isfinite(depth) & (depth > 0.05)
        if confidence is not None and confidence.shape == depth.shape and confidence.any():
            valid &= confidence >= 1
        if self._grid is None or self._grid[0].shape != depth.shape:
            self._grid = np.meshgrid(np.arange(depth.shape[1], dtype=np.float32),
                                     np.arange(depth.shape[0], dtype=np.float32))
            self.plane = None
        if self.plane is None or self._n % self.refit_every == 0:
            table = valid.copy()
            if self._last_height is not None and self.plane is not None:
                table &= ~(self._last_height > 0.5 * RAISED_M)
            p = fit_inverse_plane(depth, table)
            if p is not None:
                self.plane = p
        self._n += 1
        if self.plane is None:
            return None
        a, b, c = self.plane
        u, v = self._grid
        inv = a * u + b * v + c
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(valid & (inv > 1e-6), 1.0 / inv - depth, np.nan).astype(np.float32)
        self._last_height = h
        return h
