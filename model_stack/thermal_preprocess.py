"""FLIR One Pro thermal preprocessing — bridges domain gap to training data.

The FLIR One Pro output differs from the FLIR ADAS v2 training data in several ways:
  1. MSX mode blends visible edges onto thermal (must be stripped or disabled)
  2. May output in color palette (Ironbow/Rainbow) instead of grayscale
  3. Native resolution is 160×120 upscaled to 640×480 (blurry)
  4. Dynamic range may differ from training distribution (mean~130, std~60)

This module normalizes the FLIR One Pro output to match training data distribution.

Usage:
    from thermal_preprocess import FLIROneProPreprocessor
    preprocessor = FLIROneProPreprocessor()

    # In your capture loop:
    ret, raw_thermal = thermal_cap.read()
    processed = preprocessor.process(raw_thermal)
    # processed is now grayscale 3-channel BGR matching training distribution
"""

import numpy as np
import cv2


class FLIROneProPreprocessor:
    """Preprocess FLIR One Pro thermal frames to match training data distribution."""

    def __init__(self, target_mean=130.0, target_std=60.0, clahe_clip=2.0):
        """
        Args:
            target_mean: target pixel mean to match training data (~130 for FLIR ADAS)
            target_std: target pixel std to match training data (~60 for FLIR ADAS)
            clahe_clip: CLAHE clip limit for contrast enhancement
        """
        self.target_mean = target_mean
        self.target_std = target_std
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))

    def process(self, frame):
        """Process a raw FLIR One Pro frame to match training distribution.

        Args:
            frame: raw BGR frame from cv2.VideoCapture (may be color/MSX/grayscale)

        Returns:
            processed: grayscale thermal as 3-channel BGR, histogram-matched
        """
        if frame is None:
            return None

        # Step 1: Convert to single-channel grayscale
        gray = self._to_grayscale(frame)

        # Step 2: Remove MSX artifacts if present
        gray = self._remove_msx(gray)

        # Step 3: Enhance contrast (CLAHE) — compensates for FLIR One Pro's
        # limited 160×120 native resolution and compressed dynamic range
        gray = self.clahe.apply(gray)

        # Step 4: Histogram normalization to match training distribution
        gray = self._normalize_histogram(gray)

        # Step 5: Convert back to 3-channel BGR (model expects 3-channel input)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _to_grayscale(self, frame):
        """Convert any input format to single-channel grayscale."""
        if len(frame.shape) == 2:
            return frame

        # Check if it's a color palette (Ironbow, Rainbow, etc.)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].mean()

        if saturation > 20:
            # Color palette detected — extract intensity (value channel)
            # This handles Ironbow, Rainbow, Lava, Arctic, etc.
            gray = hsv[:, :, 2]
        else:
            # Already grayscale or near-grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return gray

    def _remove_msx(self, gray):
        """Remove MSX (Multi-Spectral Dynamic Imaging) edge overlay artifacts.

        MSX blends high-frequency edges from the visible camera onto thermal.
        These appear as sharp edge artifacts that don't exist in pure thermal.
        We use a bilateral filter to smooth edges while preserving thermal blobs.
        """
        # Bilateral filter: smooths sharp MSX edges but keeps thermal gradients
        # d=9: diameter of pixel neighborhood
        # sigmaColor=75: color space filter sigma (larger = more smoothing)
        # sigmaSpace=75: coordinate space sigma
        smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        return smoothed

    def _normalize_histogram(self, gray):
        """Normalize pixel distribution to match FLIR ADAS v2 training data.

        Training data stats: mean ≈ 130, std ≈ 60, range 0-255
        FLIR One Pro may have: mean ≈ 100-180, std ≈ 20-40, compressed range
        """
        gray = gray.astype(np.float32)

        # Standardize
        src_mean = gray.mean()
        src_std = gray.std()

        if src_std < 1.0:
            # Flat image (e.g., lens cap on) — return mid-gray
            return np.full_like(gray, self.target_mean, dtype=np.uint8)

        # Match target distribution
        normalized = (gray - src_mean) / src_std * self.target_std + self.target_mean

        # Clip to valid range
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        return normalized


class ThermalReader:
    """Complete thermal camera reader with preprocessing for FLIR One Pro."""

    def __init__(self, device="/dev/video3", preprocess=True):
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            # Try numeric
            try:
                self.cap = cv2.VideoCapture(int(device))
            except ValueError:
                pass
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open thermal camera: {device}")

        self.preprocessor = FLIROneProPreprocessor() if preprocess else None

        # Read first frame to detect format
        ret, frame = self.cap.read()
        if ret:
            print(f"[Thermal] Device: {device}")
            print(f"[Thermal] Raw frame: {frame.shape}, dtype={frame.dtype}")
            print(f"[Thermal] Mean={frame.mean():.1f}, Std={frame.astype(float).std():.1f}")
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1].mean()
            print(f"[Thermal] Saturation={sat:.1f} ({'COLOR PALETTE' if sat > 20 else 'GRAYSCALE'})")
            if self.preprocessor:
                processed = self.preprocessor.process(frame)
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                print(f"[Thermal] After preprocessing: mean={gray.mean():.1f}, std={gray.std():.1f}")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        if self.preprocessor:
            frame = self.preprocessor.process(frame)

        return True, frame

    def release(self):
        self.cap.release()


# ═══════════════════════════════════════════════════════════════════════
# Diagnostic tool — run this to check your FLIR One Pro output
# ═══════════════════════════════════════════════════════════════════════

def diagnose_thermal(device="/dev/video3"):
    """Run diagnostics on the thermal camera to identify issues."""
    import sys

    print("=" * 60)
    print("FLIR One Pro Diagnostic Tool")
    print("=" * 60)

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"FAILED: Cannot open {device}")
        print("Try: /dev/video0, /dev/video1, /dev/video2, /dev/video3")
        print("Run: v4l2-ctl --list-devices")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret:
        print("FAILED: Can read device but no frame returned")
        cap.release()
        sys.exit(1)

    print(f"\nRaw frame properties:")
    print(f"  Shape: {frame.shape}")
    print(f"  Dtype: {frame.dtype}")
    print(f"  Min/Max: {frame.min()} / {frame.max()}")
    print(f"  Mean: {frame.mean():.1f}")
    print(f"  Std: {frame.astype(float).std():.1f}")

    # Check if color or grayscale
    if len(frame.shape) == 3:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].mean()
        print(f"  Saturation: {sat:.1f}")

        if sat > 20:
            print(f"\n  WARNING: Color palette detected (saturation={sat:.1f})")
            print(f"  The FLIR One Pro is outputting in a color palette (Ironbow/Rainbow)")
            print(f"  SOLUTION: Switch to White-Hot or Black-Hot palette in FLIR app")
            print(f"  OR: Our preprocessor will auto-convert from color to grayscale")
        else:
            print(f"\n  OK: Grayscale output detected")

        # Check for MSX
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.mean() / 255.0
        print(f"  Edge density: {edge_density:.3f}")
        if edge_density > 0.05:
            print(f"  WARNING: High edge density suggests MSX mode is ON")
            print(f"  MSX blends visible camera edges onto thermal")
            print(f"  SOLUTION: Disable MSX in FLIR app, or our preprocessor will smooth it")
        else:
            print(f"  OK: No MSX artifacts detected")

    # Check resolution
    h, w = frame.shape[:2]
    print(f"\n  Resolution: {w}×{h}")
    if w * h < 200000:
        print(f"  WARNING: Low resolution ({w}×{h})")
        print(f"  FLIR One Pro native is 160×120, usually upscaled to 640×480")

    # Compare to training distribution
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    print(f"\n  Comparison to training data:")
    print(f"    Training: mean≈130, std≈60, range 0-255")
    print(f"    Your cam: mean={gray.mean():.1f}, std={gray.std():.1f}, range {gray.min()}-{gray.max()}")
    mean_diff = abs(gray.mean() - 130)
    std_diff = abs(gray.std() - 60)
    if mean_diff > 30 or std_diff > 25:
        print(f"  WARNING: Distribution mismatch (mean diff={mean_diff:.0f}, std diff={std_diff:.0f})")
        print(f"  Our preprocessor will histogram-match to training distribution")
    else:
        print(f"  OK: Distribution is close to training data")

    # Test with preprocessor
    print(f"\nTesting preprocessor...")
    preprocessor = FLIROneProPreprocessor()
    processed = preprocessor.process(frame)
    proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    print(f"  After preprocessing: mean={proc_gray.mean():.1f}, std={proc_gray.std():.1f}")
    print(f"  Target: mean≈130, std≈60")

    # Save diagnostic images
    cv2.imwrite("/tmp/thermal_raw.png", frame)
    cv2.imwrite("/tmp/thermal_processed.png", processed)
    print(f"\n  Saved: /tmp/thermal_raw.png (raw from camera)")
    print(f"  Saved: /tmp/thermal_processed.png (after preprocessing)")
    print(f"  Compare these visually to FLIR ADAS v2 training images")

    cap.release()
    print(f"\nDiagnostic complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video3")
    args = parser.parse_args()
    diagnose_thermal(args.device)
