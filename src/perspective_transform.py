"""Visualize a simple bird's-eye perspective transformation."""

from pathlib import Path

import cv2
import numpy as np


def main():
    # Resolve paths from the project directory.
    project_root = Path(__file__).resolve().parent.parent
    image_path = project_root / "test_images" / "solidYellowCurve.jpg"
    output_dir = project_root / "output_images"

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: could not read input image: {image_path}")
        return

    height, width = image.shape[:2]

    # Adjust these road points after inspecting the saved visualization.
    # Order: top-left, top-right, bottom-right, bottom-left.
    source_points = np.float32([
        (width * 0.45, height * 0.60),
        (width * 0.55, height * 0.60),
        # Experimental adjustment to better align with the right lane marking.
        (width * 0.82, height * 0.95),
        (width * 0.20, height * 0.95),
    ])

    # Map the trapezoid to a rectangle using the same point order.
    destination_points = np.float32([
        (width * 0.25, height * 0.05),
        (width * 0.75, height * 0.05),
        (width * 0.75, height * 0.95),
        (width * 0.25, height * 0.95),
    ])

    # Compute and apply the transform, keeping the original image size.
    matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    birds_eye = cv2.warpPerspective(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR
    )

    # Outline the source trapezoid in green and mark its corners in red.
    points_image = image.copy()
    drawing_points = source_points.astype(np.int32)
    cv2.polylines(points_image, [drawing_points], True, (0, 255, 0), 2)
    for x, y in drawing_points:
        cv2.circle(points_image, (int(x), int(y)), 6, (0, 0, 255), -1)

    points_path = output_dir / "solidYellowCurve_perspective_points.jpg"
    birds_eye_path = output_dir / "solidYellowCurve_birds_eye.jpg"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        points_saved = cv2.imwrite(str(points_path), points_image)
        birds_eye_saved = cv2.imwrite(str(birds_eye_path), birds_eye)
    except (OSError, cv2.error) as error:
        print(f"Error: could not save output images: {error}")
        return

    if points_saved:
        print(f"Perspective points image saved to {points_path}")
    else:
        print(f"Error: could not save perspective points image: {points_path}")

    if birds_eye_saved:
        print(f"Bird's-eye image saved to {birds_eye_path}")
    else:
        print(f"Error: could not save bird's-eye image: {birds_eye_path}")


if __name__ == "__main__":
    main()
