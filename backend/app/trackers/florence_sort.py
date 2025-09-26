from __future__ import annotations

import numpy as np

from .sort import Sort as _BaseSort


class FlorenceSort(_BaseSort):
    """Thin wrapper around base SORT to mirror the florence2 pipeline expectations.

    - Accepts detections as Nx4 or Nx5; auto-adds score=1.0 when omitted
    - Ensures float32 dtype and robust handling of empty/None inputs
    - Returns the same output format as base SORT: Nx5 [x1,y1,x2,y2,track_id]
    """

    def update(self, dets: np.ndarray | list | None = None):  # type: ignore[override]
        if dets is None:
            dets_arr = np.empty((0, 5), dtype=np.float32)
        else:
            dets_arr = np.asarray(dets, dtype=np.float32)
            if dets_arr.size == 0:
                dets_arr = np.empty((0, 5), dtype=np.float32)
            elif dets_arr.ndim == 2 and dets_arr.shape[1] == 4:
                ones = np.ones((dets_arr.shape[0], 1), dtype=np.float32)
                dets_arr = np.concatenate([dets_arr, ones], axis=1)
            elif dets_arr.ndim != 2 or dets_arr.shape[1] != 5:
                dets_arr = dets_arr.reshape(-1, 5).astype(np.float32, copy=False)

        return super().update(dets_arr)


