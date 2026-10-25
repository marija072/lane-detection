"""Calibrate the camera using chessboard images in camera_cal/."""

from pathlib import Path
from datetime import datetime, timezone
import shutil

import cv2
import numpy as np


def main():
    pattern_size = (9, 6)  # Columns and rows of internal corners.
    image_size = (1280, 720)  # Required width and height; never resize inputs.
    calibration_folder = Path(__file__).resolve().parent.parent / "camera_cal"
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(
        path for path in calibration_folder.glob("*")
        if path.is_file() and path.suffix.lower() in image_extensions
    )

    if not image_paths:
        print("No calibration images found in camera_cal/.")
        return

    # Create flat chessboard coordinates, with one square as the unit length.
    object_grid = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    object_grid[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    object_points = []
    image_points = []
    accepted_images = []
    skipped_images = []
    skipped_reasons = []

    def skip(image_path, reason):
        skipped_images.append(image_path.name)
        skipped_reasons.append(reason)
        print(f"Skipping {image_path.name}: {reason}")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    for image_path in image_paths:
        try:
            image = cv2.imread(str(image_path))
        except cv2.error as error:
            skip(image_path, f"could not read image: {error}")
            continue
        if image is None:
            skip(image_path, "could not read image")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_size = (gray.shape[1], gray.shape[0])
        if current_size != image_size:
            skip(image_path, f"dimensions {current_size[0]}x{current_size[1]}; required 1280x720")
            continue

        # Detect corners and refine their positions to subpixel accuracy.
        try:
            found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
            if found:
                refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        except cv2.error as error:
            skip(image_path, f"corner detection or refinement failed: {error}")
            continue
        if not found:
            skip(image_path, "chessboard corners not detected")
            continue

        object_points.append(object_grid.copy())
        image_points.append(refined_corners)
        accepted_images.append(image_path.name)
        print(f"Detected corners in {image_path.name}.")

    print("\nCamera calibration summary")
    print(f"Images found: {len(image_paths)}")
    print(f"Images used: {len(image_points)}")
    print(f"Images skipped: {len(image_paths) - len(image_points)}")
    print(f"Accepted images: {', '.join(accepted_images) or 'none'}")
    for name, reason in zip(skipped_images, skipped_reasons):
        print(f"Skipped: {name}: {reason}")
    if not image_points:
        print("Calibration could not be performed: no valid chessboards detected.")
        return

    if len(image_points) < 3:
        print("Warning: use more images from different angles for reliable calibration.")

    # Estimate the camera's intrinsic matrix and lens distortion.
    try:
        rms_error, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(
            object_points, image_points, image_size, None, None
        )
    except cv2.error as error:
        print(f"Calibration failed: {error}")
        return

    print(f"Chessboard internal corners: {pattern_size[0]} x {pattern_size[1]}")
    print(f"Image size (width x height): {image_size[0]} x {image_size[1]}")
    print(f"RMS reprojection error: {rms_error:.4f} pixels")
    print("Camera matrix:")
    print(camera_matrix)
    print("Distortion coefficients (k1, k2, p1, p2, k3):")
    print(distortion_coefficients.ravel())

    # Save the calibration arrays for use in later processing steps.
    output_folder = calibration_folder.parent / "calibration_data"
    output_path = output_folder / "camera_calibration.npz"
    run_time = datetime.now(timezone.utc).isoformat()
    try:
        output_folder.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            # Preserve the previous file before replacing any saved parameters.
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup_path = output_folder / f"camera_calibration_backup_{stamp}.npz"
            shutil.copy2(output_path, backup_path)
            print(f"Previous calibration backed up to {backup_path}")
        np.savez(
            output_path,
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
            image_size=np.array(image_size, dtype=np.int32),
            rms_reprojection_error=rms_error,
            chessboard_dimensions=np.array(pattern_size, dtype=np.int32),
            accepted_images=np.array(accepted_images, dtype=str),
            skipped_images=np.array(skipped_images, dtype=str),
            skipped_reasons=np.array(skipped_reasons, dtype=str),
            calibration_time_utc=run_time,
            project_assumptions=np.array([
                "Calibration, road images, and videos use the same physical camera (assumed).",
                "960x540 road inputs are uniform resizes of the 1280x720 full view (assumed).",
            ]),
        )
    except (OSError, ValueError) as error:
        print(f"Could not back up or save calibration: {error}")
        return
    print("Calibration results saved to calibration_data/camera_calibration.npz")


if __name__ == "__main__":
    main()
