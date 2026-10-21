"""Correct lens distortion in a calibration image."""

from pathlib import Path

import cv2
import numpy as np


def main():
    # Resolve paths from the project directory.
    project_root = Path(__file__).resolve().parent.parent
    calibration_path = project_root / "calibration_data" / "camera_calibration.npz"
    image_path = project_root / "camera_cal" / "calibration10.jpg"
    output_path = project_root / "output_images" / "calibration10_undistorted.jpg"

    if not calibration_path.is_file():
        print(f"Error: calibration file not found: {calibration_path}")
        return
    if not image_path.is_file():
        print(f"Error: input image not found: {image_path}")
        return

    # Load the arrays saved by the camera calibration script.
    try:
        with np.load(calibration_path, allow_pickle=False) as calibration_data:
            camera_matrix = calibration_data["camera_matrix"]
            distortion_coefficients = calibration_data["distortion_coefficients"]
    except (OSError, ValueError, KeyError) as error:
        print(f"Error: could not load calibration data: {error}")
        return

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: could not read input image: {image_path}")
        return

    # Correct lens distortion using the saved camera parameters.
    try:
        undistorted_image = cv2.undistort(
            image, camera_matrix, distortion_coefficients
        )
    except cv2.error as error:
        print(f"Error: could not undistort image: {error}")
        return

    # Create the output directory and save the corrected image.
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        saved = cv2.imwrite(str(output_path), undistorted_image)
    except (OSError, cv2.error) as error:
        print(f"Error: could not save corrected image: {error}")
        return

    if not saved:
        print(f"Error: could not save corrected image: {output_path}")
        return

    print("Corrected image saved to output_images/calibration10_undistorted.jpg")

    # Add a label strip above the full-size images.
    comparison = cv2.hconcat([image, undistorted_image])
    comparison = cv2.copyMakeBorder(
        comparison, 60, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    for label, x in [("Original", 20), ("Undistorted", image.shape[1] + 20)]:
        cv2.putText(
            comparison, label, (x, 40), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 0), 2, cv2.LINE_AA
        )

    comparison_path = output_path.parent / "calibration10_comparison.jpg"
    try:
        saved = cv2.imwrite(str(comparison_path), comparison)
    except (OSError, cv2.error) as error:
        print(f"Error: could not save comparison image: {error}")
        return

    if not saved:
        print(f"Error: could not save comparison image: {comparison_path}")
        return

    print("Comparison image saved to output_images/calibration10_comparison.jpg")


if __name__ == "__main__":
    main()
