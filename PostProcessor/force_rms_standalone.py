"""Force RMS only. Same arithmetic as squish_core.force_rms.

  RMS      = SD of linearly detrended force in [t0, t1]
  RMS_corr = sqrt(max(0, RMS^2 - 0.166^2))

The 0.166 uN floor is the median day-8.5 hold-window force RMS.
Windows with fewer than 50 samples return None.

Typical hold window: 35 to 55 s (30 s-ramp rounds).
"""
import numpy as np
import pandas as pd
from scipy.signal import detrend

RMS0_FORCE_UN = 0.166
MIN_SAMPLES = 50
DEFAULT_WINDOW = (35.0, 55.0)


def load_force(path):
    """Time (s) and force (uN) from one *Data.csv. Col 2 = time in ms, col 3 = force."""
    df = pd.read_csv(path, header=0, encoding="utf-8", encoding_errors="replace")
    t = df.iloc[:, 2].astype(float).to_numpy() / 1000.0
    F = df.iloc[:, 3].astype(float).to_numpy()
    return t, F


def detrended_rms(x):
    """SD of the linear-detrended signal (= RMS once the mean is zero)."""
    return float(np.std(detrend(np.asarray(x, float), type="linear")))


def quadrature_subtract(rms_raw, floor=RMS0_FORCE_UN):
    return float(np.sqrt(max(0.0, float(rms_raw) ** 2 - float(floor) ** 2)))


def force_rms(t, F, t0=DEFAULT_WINDOW[0], t1=DEFAULT_WINDOW[1],
              correct=True, min_samples=MIN_SAMPLES):
    """Noise-corrected force RMS (uN) in [t0, t1], or None if the window is too short."""
    mask = (t >= t0) & (t <= t1)
    if np.count_nonzero(mask) < min_samples:
        return None
    raw = detrended_rms(F[mask])
    return quadrature_subtract(raw) if correct else raw


def force_rms_from_csv(path, t0=DEFAULT_WINDOW[0], t1=DEFAULT_WINDOW[1], correct=True):
    t, F = load_force(path)
    return force_rms(t, F, t0, t1, correct=correct)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python force_rms_standalone.py /path/to/recordingData.csv")
        raise SystemExit(1)
    val = force_rms_from_csv(sys.argv[1])
    print(f"force RMS (35-55 s, noise-corrected) = {val} uN")
