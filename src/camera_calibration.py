"""Calibrate the camera using chessboard images in camera_cal/."""

from pathlib import Path

import cv2
import numpy as np


def main():
    pattern_size = (9, 6)  # Columns and rows of internal corners.
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
    image_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping {image_path.name}: could not read image.")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_size = (gray.shape[1], gray.shape[0])
        if image_size is not None and current_size != image_size:
            print(f"Skipping {image_path.name}: image size differs from other images.")
            continue

        # Detect corners and refine their positions to subpixel accuracy.
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if not found:
            print(f"Skipping {image_path.name}: chessboard corners not detected.")
            continue

        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_grid.copy())
        image_points.append(refined_corners)
        image_size = current_size
        print(f"Detected corners in {image_path.name}.")

    print("\nCamera calibration summary")
    print(f"Images found: {len(image_paths)}")
    print(f"Images used: {len(image_points)}")
    print(f"Images skipped: {len(image_paths) - len(image_points)}")
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
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / "camera_calibration.npz"
    np.savez(
        output_path,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
    )
    print("Calibration results saved to calibration_data/camera_calibration.npz")


if __name__ == "__main__":
    main()
