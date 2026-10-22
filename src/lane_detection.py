"""Detect straight line segments in a road image."""

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

    # Find edges across the entire image.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 170)

    # Define a road trapezoid with a wide bottom and narrow top.
    height, width = edges.shape
    roi_points = np.array([[
        (int(width * 0.10), height - 1),
        (int(width * 0.45), int(height * 0.60)),
        (int(width * 0.55), int(height * 0.60)),
        (int(width * 0.90), height - 1),
    ]], dtype=np.int32)

    # Fill the mask and keep only edges inside the trapezoid.
    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, roi_points, 255)
    roi_edges = cv2.bitwise_and(edges, roi_mask)

    # Detect straight line segments in the edge image.
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=100,
        maxLineGap=10,
    )

    # Draw each segment in red (OpenCV uses BGR colors).
    line_image = image.copy()
    if lines is None:
        print("No line segments detected.")
    else:
        for line in lines.reshape(-1, 4):
            x1, y1, x2, y2 = map(int, line)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Detect segments inside the ROI using the same Hough parameters.
    roi_lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=10,
    )

    # Draw ROI segments on a fresh copy of the original image.
    roi_line_image = image.copy()
    if roi_lines is None:
        print("No line segments detected inside the ROI.")
    else:
        for line in roi_lines.reshape(-1, 4):
            x1, y1, x2, y2 = map(int, line)
            cv2.line(roi_line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Filter ROI segments by slope on a separate image.
    filtered_line_image = image.copy()
    left_count = 0
    right_count = 0
    left_points = []
    right_points = []
    if roi_lines is not None:
        for line in roi_lines.reshape(-1, 4):
            x1, y1, x2, y2 = map(int, line)
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:
                continue

            # Negative slopes are left candidates; positive slopes are right.
            if slope < 0:
                color = (0, 255, 0)
                left_count += 1
                left_points.extend([(x1, y1), (x2, y2)])
            else:
                color = (255, 0, 0)
                right_count += 1
                right_points.extend([(x1, y1), (x2, y2)])
            cv2.line(filtered_line_image, (x1, y1), (x2, y2), color, 2)

    print(f"Accepted left segments: {left_count}")
    print(f"Accepted right segments: {right_count}")

    # Fit x = a*y + b for each boundary using its accepted endpoints.
    fitted_lane_image = image.copy()
    y_bottom = int(height * 0.95)
    y_top = int(roi_points[0, :, 1].min())
    for points, color, side in [
        (left_points, (0, 255, 0), "left"),
        (right_points, (255, 0, 0), "right"),
    ]:
        if len(points) < 2:
            print(f"Not enough points to fit the {side} boundary.")
            continue

        x_coordinates, y_coordinates = np.array(points, dtype=float).T
        if np.ptp(y_coordinates) < 1:
            print(f"Not enough y variation to fit the {side} boundary.")
            continue

        try:
            a, b = np.polyfit(y_coordinates, x_coordinates, 1)
        except (ValueError, np.linalg.LinAlgError) as error:
            print(f"Could not fit the {side} boundary: {error}")
            continue

        # Evaluate the fitted line at the bottom and the top of the ROI.
        x_bottom = a * y_bottom + b
        x_top = a * y_top + b
        if not np.all(np.isfinite([x_bottom, x_top])):
            print(f"Invalid fitted coordinates for the {side} boundary.")
            continue

        cv2.line(
            fitted_lane_image,
            (int(x_bottom), y_bottom),
            (int(x_top), y_top),
            color,
            2,
        )

    # Save all results, even when no lines were found.
    edges_path = output_dir / "solidYellowCurve_edges.jpg"
    hough_path = output_dir / "solidYellowCurve_hough.jpg"
    roi_edges_path = output_dir / "solidYellowCurve_roi_edges.jpg"
    roi_hough_path = output_dir / "solidYellowCurve_roi_hough.jpg"
    filtered_hough_path = output_dir / "solidYellowCurve_filtered_hough.jpg"
    fitted_lanes_path = output_dir / "solidYellowCurve_fitted_lanes.jpg"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        edges_saved = cv2.imwrite(str(edges_path), edges)
        hough_saved = cv2.imwrite(str(hough_path), line_image)
        roi_edges_saved = cv2.imwrite(str(roi_edges_path), roi_edges)
        roi_hough_saved = cv2.imwrite(str(roi_hough_path), roi_line_image)
        filtered_hough_saved = cv2.imwrite(str(filtered_hough_path), filtered_line_image)
        fitted_lanes_saved = cv2.imwrite(str(fitted_lanes_path), fitted_lane_image)
    except (OSError, cv2.error) as error:
        print(f"Error: could not save output images: {error}")
        return

    if edges_saved:
        print(f"Edge image saved to {edges_path}")
    else:
        print(f"Error: could not save edge image: {edges_path}")

    if hough_saved:
        print(f"Line image saved to {hough_path}")
    else:
        print(f"Error: could not save line image: {hough_path}")

    if roi_edges_saved:
        print(f"ROI edge image saved to {roi_edges_path}")
    else:
        print(f"Error: could not save ROI edge image: {roi_edges_path}")

    if roi_hough_saved:
        print(f"ROI line image saved to {roi_hough_path}")
    else:
        print(f"Error: could not save ROI line image: {roi_hough_path}")

    if filtered_hough_saved:
        print(f"Filtered line image saved to {filtered_hough_path}")
    else:
        print(f"Error: could not save filtered line image: {filtered_hough_path}")

    if fitted_lanes_saved:
        print(f"Fitted lane image saved to {fitted_lanes_path}")
    else:
        print(f"Error: could not save fitted lane image: {fitted_lanes_path}")


if __name__ == "__main__":
    main()
