"""Detect lane candidates in a road image and save diagnostic visualizations."""

import argparse
from pathlib import Path

import cv2
import numpy as np


# Enable this experiment to estimate perspective points from road markings.
USE_ADAPTIVE_PERSPECTIVE = False


def estimate_source_points(mask):
    """Return a supported road trapezoid, or a reason to use fixed points."""
    height, width = mask.shape
    y_top = height * 0.60
    y_bottom = height * 0.95
    try:
        segments = cv2.HoughLinesP(
            mask, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=10
        )
    except cv2.error as error:
        return None, f"Hough estimation failed: {error}"
    if segments is None:
        return None, "no supporting line segments"

    groups = {"Left": [], "Right": []}
    for segment in segments.reshape(-1, 4):
        x1, y1, x2, y2 = map(int, segment)
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.3:
            continue
        # Estimate x at each selected height using this segment's line.
        top_x = x1 + (y_top - y1) / slope
        bottom_x = x1 + (y_bottom - y1) / slope
        side = "Left" if slope < 0 else "Right"
        if (side == "Left" and bottom_x >= width / 2
                or side == "Right" and bottom_x <= width / 2):
            continue
        groups[side].append((top_x, bottom_x, y1, y2))

    boundaries = {}
    for side, group in groups.items():
        if len(group) < 3:
            return None, f"too few {side.lower()} supporting segments"
        values = np.array(group)
        # Require vertical support near both heights to limit extrapolation.
        if (values[:, 2:].min() > y_top + height * 0.05
                or values[:, 2:].max() < y_bottom - height * 0.05):
            return None, f"insufficient {side.lower()} vertical support"
        positions = values[:, :2]
        estimate = np.median(positions, axis=0)
        # Reject estimates whose segments disagree substantially at either height.
        if np.any(np.percentile(np.abs(positions - estimate), 75, axis=0) > width * 0.08):
            return None, f"unstable {side.lower()} boundary estimates"
        boundaries[side] = estimate

    left_top, left_bottom = boundaries["Left"]
    right_top, right_bottom = boundaries["Right"]
    top_width = right_top - left_top
    bottom_width = right_bottom - left_bottom
    points = np.float32([
        (left_top, y_top), (right_top, y_top),
        (right_bottom, y_bottom), (left_bottom, y_bottom),
    ])
    if not np.all(np.isfinite(points)):
        return None, "non-finite source coordinates"
    if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width):
        return None, "source points outside the image"
    # Keep a clear, widening road trapezoid with correctly ordered boundaries.
    if not (width * 0.05 <= top_width <= width * 0.40
            and width * 0.25 <= bottom_width <= width * 0.95
            and bottom_width > top_width * 1.2
            and left_bottom < left_top and right_bottom > right_top):
        return None, "implausible trapezoid geometry"
    return points, "supported lane segments"


def radius_of_curvature(coefficients, y):
    """Return curvature radius in warped-image pixels, not meters."""
    a, b, _ = coefficients
    # A nearly straight boundary has an effectively infinite radius.
    if abs(a) < 1e-10:
        return float("inf")

    # Use the first and second derivatives of x(y) in the curvature formula.
    return (1 + (2 * a * y + b) ** 2) ** 1.5 / abs(2 * a)


def process_frame(frame, diagnostics=False, verbose=False, *,
                  color_gated_canny=False, dilation_kernel_size=5):
    """Return (BGR result, metadata, diagnostics dict) without file I/O.

    A rejected detection returns an unchanged copy of the input frame.
    Curvature and offset remain in warped-image pixels, not meters.
    Experimental color gating keeps Canny edges only near HSV-selected pixels.
    """
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0):
        raise ValueError("frame must be a nonempty uint8 BGR image")
    image = frame

    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # OpenCV HSV ranges: H = 0-179, S and V = 0-255.
    # Adjust these initial thresholds for different lighting conditions.
    yellow_lower_hsv = np.array([15, 80, 80], dtype=np.uint8)
    yellow_upper_hsv = np.array([35, 255, 255], dtype=np.uint8)
    white_lower_hsv = np.array([0, 0, 200], dtype=np.uint8)
    white_upper_hsv = np.array([179, 50, 255], dtype=np.uint8)

    # Yellow has a limited hue range; white is bright with low saturation.
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv_image, yellow_lower_hsv, yellow_upper_hsv)
    white_mask = cv2.inRange(hsv_image, white_lower_hsv, white_upper_hsv)

    # Keep pixels selected by either color mask.
    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
    if diagnostics:
        color_filtered = cv2.bitwise_and(image, image, mask=combined_mask)

    # Adjust these fractions to cover the road, allowing room for curves.
    height, width = combined_mask.shape
    roi_points = np.array([[
        (int(width * 0.35), int(height * 0.55)),
        (int(width * 0.65), int(height * 0.55)),
        (int(width * 0.95), int(height * 0.95)),
        # Widen the lower-left edge to include the yellow marking.
        (int(width * 0.00), int(height * 0.95)),
    ]], dtype=np.int32)

    # Fill the road trapezoid and keep color pixels only inside it.
    roi_mask = np.zeros_like(combined_mask)
    cv2.fillPoly(roi_mask, roi_points, 255)
    if diagnostics:
        color_roi_mask = cv2.bitwise_and(combined_mask, roi_mask)

    # Find edges to complement the pixels selected by color.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_edges = cv2.Canny(blurred, 100, 170)

    # Keep color or edge information, then restrict it to the road ROI.
    edges_to_combine = canny_edges
    if color_gated_canny:
        if (not isinstance(dilation_kernel_size, int)
                or dilation_kernel_size < 1 or dilation_kernel_size % 2 == 0):
            raise ValueError("dilation_kernel_size must be a positive odd integer")
        # Dilate only the edge-selection gate; keep the HSV mask itself unchanged.
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), dtype=np.uint8)
        color_gate = cv2.dilate(combined_mask, kernel, iterations=1)
        edges_to_combine = cv2.bitwise_and(canny_edges, color_gate)
    hsv_canny_mask = cv2.bitwise_or(combined_mask, edges_to_combine)
    hsv_canny_roi = cv2.bitwise_and(hsv_canny_mask, roi_mask)

    # Keep these fixed points as the default and adaptive fallback.
    # Order: top-left, top-right, bottom-right, bottom-left.
    source_points = np.float32([
        (width * 0.45, height * 0.60),
        (width * 0.55, height * 0.60),
        (width * 0.82, height * 0.95),
        (width * 0.20, height * 0.95),
    ])
    if USE_ADAPTIVE_PERSPECTIVE:
        adaptive_points, perspective_reason = estimate_source_points(hsv_canny_roi)
        if adaptive_points is not None:
            source_points = adaptive_points
            log(f"Perspective for frame: adaptive ({perspective_reason}).")
        else:
            log(f"Perspective for frame: fixed fallback ({perspective_reason}).")
    else:
        log(f"Perspective for frame: fixed (adaptive selection disabled).")

    destination_points = np.float32([
        (width * 0.25, height * 0.05),
        (width * 0.75, height * 0.05),
        (width * 0.75, height * 0.95),
        (width * 0.25, height * 0.95),
    ])

    if diagnostics:
        # Show the existing source trapezoid and its ordered corners.
        perspective_source_image = image.copy()
        drawing_points = source_points.astype(np.int32)
        cv2.polylines(perspective_source_image, [drawing_points], True, (0, 255, 0), 3)
        point_labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
        for (x, y), label in zip(drawing_points, point_labels):
            point = (int(x), int(y))
            cv2.circle(perspective_source_image, point, 6, (0, 0, 255), -1)
            cv2.putText(
                perspective_source_image, label, (int(x) + 8, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA
            )

    # Warp the mask to a bird's-eye view without blending binary values.
    matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    hsv_canny_birds_eye = cv2.warpPerspective(
        hsv_canny_roi, matrix, (width, height), flags=cv2.INTER_NEAREST
    )

    # Sum white-pixel values in each column of the mask's bottom half.
    histogram = np.sum(hsv_canny_birds_eye[height // 2:, :], axis=0)
    midpoint = width // 2
    log(f"Image width: {width}")
    peak_positions = {}
    # These peaks are initial candidates, not confirmed lane positions.
    for side, start, end in [("Left", 0, midpoint), ("Right", midpoint, width)]:
        half_histogram = histogram[start:end]
        if half_histogram.size == 0 or half_histogram.max() == 0:
            log(f"{side} peak candidate: no white pixels")
        else:
            peak_position = start + int(np.argmax(half_histogram))
            peak_positions[side] = peak_position
            log(f"{side} peak candidate: column {peak_position}")

    if diagnostics:
        # Mark the existing peak candidates on a color copy of the binary mask.
        histogram_peaks_image = cv2.cvtColor(hsv_canny_birds_eye, cv2.COLOR_GRAY2BGR)
        for side, color, label_y in [
            ("Left", (0, 255, 0), 25),
            ("Right", (255, 0, 0), 50),
        ]:
            if side not in peak_positions:
                continue
            peak_position = peak_positions[side]
            cv2.line(
                histogram_peaks_image, (peak_position, 0),
                (peak_position, height - 1), color, 1
            )
            label = f"{side} peak candidate"
            text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            label_x = max(0, min(peak_position + 5, width - text_width - 5))
            cv2.putText(
                histogram_peaks_image, label, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )

        # Scale the values to a dark graph while preserving column positions.
        graph_height = 300
        histogram_image = np.zeros((graph_height, width, 3), dtype=np.uint8)
        normalized_histogram = np.zeros(width, dtype=float)
        if histogram.max() > 0:
            normalized_histogram = histogram / histogram.max() * (graph_height - 1)
        graph_points = np.column_stack((
            np.arange(width),
            graph_height - 1 - normalized_histogram,
        )).astype(np.int32)
        cv2.polylines(histogram_image, [graph_points], False, (0, 255, 0), 1)

    # Search for candidate pixels in nine windows from bottom to top.
    number_of_windows = 9
    window_margin = 80
    minimum_pixels = 30
    if diagnostics:
        sliding_windows_image = cv2.cvtColor(hsv_canny_birds_eye, cv2.COLOR_GRAY2BGR)
    nonzero_y, nonzero_x = hsv_canny_birds_eye.nonzero()
    lane_pixels = {}

    for side, color in [("Left", (0, 255, 0)), ("Right", (255, 0, 0))]:
        pixel_groups = []
        if side in peak_positions:
            current_x = peak_positions[side]
            for window in range(number_of_windows):
                # Integer boundaries cover the full height without gaps.
                y_low = height - (window + 1) * height // number_of_windows
                y_high = height - window * height // number_of_windows
                x_low = max(0, current_x - window_margin)
                x_high = min(width, current_x + window_margin)
                if diagnostics:
                    cv2.rectangle(
                        sliding_windows_image, (x_low, y_low),
                        (x_high - 1, y_high - 1), color, 2
                    )

                pixel_indices = np.where(
                    (nonzero_y >= y_low) & (nonzero_y < y_high)
                    & (nonzero_x >= x_low) & (nonzero_x < x_high)
                )[0]
                pixel_groups.append(pixel_indices)

                # Keep the previous position when too few pixels are found.
                if pixel_indices.size >= minimum_pixels:
                    current_x = int(np.mean(nonzero_x[pixel_indices]))

        if pixel_groups:
            lane_indices = np.concatenate(pixel_groups)
        else:
            lane_indices = np.array([], dtype=int)
        lane_pixels[side] = (nonzero_x[lane_indices], nonzero_y[lane_indices])

    left_x, left_y = lane_pixels["Left"]
    right_x, right_y = lane_pixels["Right"]
    log(f"Collected left candidate lane pixels: {left_x.size}")
    log(f"Collected right candidate lane pixels: {right_x.size}")

    # Fit x as a quadratic function of y using the collected lane pixels.
    if diagnostics:
        polynomial_lanes_image = cv2.cvtColor(hsv_canny_birds_eye, cv2.COLOR_GRAY2BGR)
    fitted_coefficients = {}
    fitted_y_ranges = {}
    for side, x_points, y_points, color in [
        ("Left", left_x, left_y, (0, 255, 0)),
        ("Right", right_x, right_y, (255, 0, 0)),
    ]:
        valid_points = np.isfinite(x_points) & np.isfinite(y_points)
        x_points = x_points[valid_points]
        y_points = y_points[valid_points]
        if x_points.size < 3 or np.unique(y_points).size < 3:
            log(f"{side} curve skipped: not enough valid points or distinct y values.")
            continue

        try:
            coefficients, _, rank, _, _ = np.polyfit(
                y_points, x_points, 2, full=True
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            log(f"{side} curve fitting failed: {error}")
            continue
        if rank < 3 or not np.all(np.isfinite(coefficients)):
            log(f"{side} curve skipped: the polynomial fit is invalid.")
            continue

        log(f"{side} polynomial coefficients: {coefficients}")
        fitted_coefficients[side] = coefficients
        # Draw only between the lowest and highest observed candidate rows.
        y_start = int(y_points.min())
        y_end = int(y_points.max())
        fitted_y_ranges[side] = (y_start, y_end)
        log(f"{side} fitted pixel support: y={y_start} to {y_end}")
        if diagnostics:
            plot_y = np.arange(y_start, y_end + 1)
            plot_x = np.polyval(coefficients, plot_y)

            # Connect neighboring points, breaking the curve outside the image.
            previous_point = None
            for x, y in zip(plot_x, plot_y):
                if not np.isfinite(x) or x < 0 or x >= width:
                    previous_point = None
                    continue
                point = (int(x), int(y))
                if previous_point is not None:
                    cv2.line(polynomial_lanes_image, previous_point, point, color, 2)
                previous_point = point

    if diagnostics:
        # Reuse the supported green and blue curves for a separate width diagnostic.
        lane_width_image = polynomial_lanes_image.copy()
        if "Left" in fitted_coefficients and "Right" in fitted_coefficients:
            for fraction in (0.25, 0.50, 0.75):
                width_y = int(height * fraction)
                width_left_x = np.polyval(fitted_coefficients["Left"], width_y)
                width_right_x = np.polyval(fitted_coefficients["Right"], width_y)
                if not (
                    np.isfinite(width_left_x) and np.isfinite(width_right_x)
                    and 0 <= width_left_x < width
                    and 0 <= width_right_x < width
                    and width_left_x < width_right_x
                ):
                    continue

                # These sample widths are in warped-image pixels.
                pixel_width = width_right_x - width_left_x
                cv2.line(
                    lane_width_image, (int(width_left_x), width_y),
                    (int(width_right_x), width_y), (0, 255, 255), 2
                )
                label = f"{pixel_width:.1f} px"
                text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
                label_x = max(0, min(
                    int((width_left_x + width_right_x - text_width) / 2),
                    width - text_width - 1,
                ))
                cv2.putText(
                    lane_width_image, label, (label_x, max(15, width_y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
                )

    # Check geometry in pixels, not physical lane width in meters.
    if "Left" not in fitted_coefficients or "Right" not in fitted_coefficients:
        log("Warning: lane-width check skipped because a fitted boundary is unavailable.")
    else:
        for fraction in (0.25, 0.50, 0.75):
            check_y = int(height * fraction)
            check_left_x = np.polyval(fitted_coefficients["Left"], check_y)
            check_right_x = np.polyval(fitted_coefficients["Right"], check_y)
            lane_width = check_right_x - check_left_x
            log(
                f"Lane-width check: y={check_y}, left x={check_left_x:.2f}, "
                f"right x={check_right_x:.2f}, width={lane_width:.2f} pixels"
            )
            if lane_width <= 0:
                log(f"Warning: lane width is zero or negative at y={check_y}.")
            if not (0 <= check_left_x < width and 0 <= check_right_x < width):
                log(f"Warning: a fitted boundary lies outside the image at y={check_y}.")

    # Curvature uses warped-image pixels and depends on perspective and scale.
    # These radii are not physical measurements in meters.
    curvature_y = height - 1
    curvature_radii = {}
    for side in ("Left", "Right"):
        if side not in fitted_coefficients:
            log(f"{side} curvature unavailable: no fitted boundary.")
            continue
        radius = radius_of_curvature(fitted_coefficients[side], curvature_y)
        curvature_radii[side] = radius
        if np.isinf(radius):
            log(f"{side} curvature radius: infinite (pixel-based, nearly straight).")
        else:
            log(f"{side} curvature radius: {radius:.2f} pixels (not meters).")

    # Sample every image row to check the fitted geometry before overlaying it.
    lane_valid = False
    validation_reason = "both fitted boundaries are required"
    if "Left" in fitted_coefficients and "Right" in fitted_coefficients:
        sample_y = np.arange(height)
        sample_left = np.polyval(fitted_coefficients["Left"], sample_y)
        sample_right = np.polyval(fitted_coefficients["Right"], sample_y)
        sample_widths = sample_right - sample_left
        if not (np.all(np.isfinite(sample_left))
                and np.all(np.isfinite(sample_right))
                and np.all(np.isfinite(sample_widths))):
            validation_reason = "fitted boundary coordinates or widths are non-finite"
        elif np.any(sample_widths <= 0):
            validation_reason = "boundaries cross or their separation is non-positive"
        # Allow moderate variation: reject only a large ratio AND a large spread.
        # This checks warped-image geometry, without assuming a physical lane width.
        elif (sample_widths.max() > 4 * sample_widths.min()
              and np.ptp(sample_widths) > width * 0.5):
            validation_reason = "lane width varies excessively (over 4x and half the image width)"
        else:
            lane_valid = True
            validation_reason = "boundary ordering and width consistency checks passed"
    validation_status = "passed" if lane_valid else "rejected"
    log(f"Lane validation: {validation_status} - {validation_reason}.")

    # Use only overlapping observed ranges; add no extrapolated extension.
    shared_y_start = 0
    shared_y_end = -1
    support_valid = False
    # Require at least 10% of image height (and two rows) for a useful polygon.
    minimum_shared_rows = max(2, int(np.ceil(height * 0.10)))
    if "Left" in fitted_y_ranges and "Right" in fitted_y_ranges:
        shared_y_start = max(fitted_y_ranges["Left"][0], fitted_y_ranges["Right"][0])
        shared_y_end = min(fitted_y_ranges["Left"][1], fitted_y_ranges["Right"][1])
        support_valid = shared_y_end - shared_y_start + 1 >= minimum_shared_rows
        if support_valid:
            log(f"Shared lane support: y={shared_y_start} to {shared_y_end} (no extension).")

    # Build the lane area only after the pair passes geometry validation.
    lane_overlay = None
    vehicle_offset = None
    if not lane_valid:
        log(f"Lane overlay skipped: {validation_reason}.")
    elif not support_valid:
        log(
            "Lane overlay skipped: insufficient overlapping vertical pixel support "
            f"(at least {minimum_shared_rows} rows required)."
        )
    else:
        lane_area = np.zeros((height, width, 3), dtype=np.uint8)
        overlay_y = np.arange(height)
        overlay_left_x = np.polyval(fitted_coefficients["Left"], overlay_y)
        overlay_right_x = np.polyval(fitted_coefficients["Right"], overlay_y)
        valid_rows = (
            (overlay_y >= shared_y_start) & (overlay_y <= shared_y_end)
            & np.isfinite(overlay_left_x) & np.isfinite(overlay_right_x)
            & (overlay_left_x >= 0) & (overlay_left_x < width)
            & (overlay_right_x >= 0) & (overlay_right_x < width)
            & (overlay_left_x < overlay_right_x)
        )
        valid_y = overlay_y[valid_rows]
        if valid_y.size > 0:
            # Assume the camera (and vehicle) is horizontally centered.
            # Positive offset means right of lane center; negative means left.
            offset_y = int(valid_y[-1])
            lane_center_x = (overlay_left_x[offset_y] + overlay_right_x[offset_y]) / 2
            vehicle_center_x = width / 2
            offset_px = vehicle_center_x - lane_center_x
            vehicle_offset = {
                "y": offset_y,
                "lane_center": float(lane_center_x),
                "vehicle_center": vehicle_center_x,
                "pixels": float(offset_px),
            }
        polygon_drawn = False

        # Keep separate valid sections separate instead of bridging invalid rows.
        row_groups = np.split(valid_y, np.where(np.diff(valid_y) > 1)[0] + 1)
        for rows in row_groups:
            if rows.size < 2:
                continue
            left_boundary = np.column_stack((overlay_left_x[rows], rows)).astype(np.int32)
            right_boundary = np.column_stack((overlay_right_x[rows], rows)).astype(np.int32)
            polygon = np.vstack((left_boundary, right_boundary[::-1]))
            if cv2.contourArea(polygon) <= 0:
                continue
            cv2.fillPoly(lane_area, [polygon], (0, 255, 0))
            cv2.polylines(lane_area, [left_boundary], False, (0, 0, 255), 3)
            cv2.polylines(lane_area, [right_boundary], False, (255, 0, 0), 3)
            polygon_drawn = True

        if polygon_drawn:
            # Reverse the perspective transform to return to the camera view.
            inverse_matrix = cv2.getPerspectiveTransform(destination_points, source_points)
            warped_lane_area = cv2.warpPerspective(
                lane_area, inverse_matrix, (width, height)
            )
            # Blend the colored area with the original so the road stays visible.
            lane_overlay = cv2.addWeighted(image, 1.0, warped_lane_area, 0.4, 0)
        else:
            log("Lane overlay skipped: no safe lane polygon could be constructed.")

    if vehicle_offset is None:
        log("Vehicle offset unavailable: missing fits, invalid geometry, or no valid shared support.")
    else:
        log(
            f"Vehicle offset at y={vehicle_offset['y']}: "
            f"lane center={vehicle_offset['lane_center']:.2f} pixels, "
            f"vehicle center={vehicle_offset['vehicle_center']:.2f} pixels, "
            f"offset={vehicle_offset['pixels']:+.2f} pixels (+right, -left)."
        )

    if diagnostics:
        # Outline the same ROI on a copy of the original image.
        roi_overlay = image.copy()
        cv2.polylines(roi_overlay, roi_points, True, (0, 255, 0), 3)


    detection_reason = "lane overlay generated"
    if lane_overlay is None:
        if not lane_valid:
            detection_reason = validation_reason
        elif not support_valid:
            detection_reason = "insufficient overlapping vertical pixel support"
        else:
            detection_reason = "no safe lane polygon could be constructed"
    metadata = {
        "success": lane_overlay is not None,
        "detection_success": lane_overlay is not None,
        "detection_reason": detection_reason,
        "left_fit": "Left" in fitted_coefficients,
        "right_fit": "Right" in fitted_coefficients,
        "radii": curvature_radii,
        "vehicle_offset": vehicle_offset,
        "overlay": lane_overlay is not None,
        "validation": validation_status,
        "validation_reason": validation_reason,
        "support_valid": support_valid,
        "fitted_y_ranges": fitted_y_ranges,
        "curvature_y": curvature_y,
        "measurement_units": "warped-image pixels",
    }
    diagnostic_images = {}
    if diagnostics:
        outputs = [
            ("perspective_source_points", perspective_source_image),
            ("yellow_mask", yellow_mask),
            ("white_mask", white_mask),
            ("combined_mask", combined_mask),
            ("color_filtered", color_filtered),
            ("roi_mask", roi_mask),
            ("color_roi_mask", color_roi_mask),
            ("roi_overlay", roi_overlay),
            ("canny_edges", canny_edges),
            ("hsv_canny_roi", hsv_canny_roi),
            ("hsv_canny_birds_eye", hsv_canny_birds_eye),
            ("lane_histogram", histogram_image),
            ("histogram_peaks", histogram_peaks_image),
            ("sliding_windows", sliding_windows_image),
            ("polynomial_lanes", polynomial_lanes_image),
            ("lane_width_visualization", lane_width_image),
        ]
        if lane_overlay is not None:
            outputs.append(("lane_overlay", lane_overlay))


        diagnostic_images = dict(outputs)
    result_frame = lane_overlay if lane_overlay is not None else image.copy()
    return result_frame, metadata, diagnostic_images


def process_image(image_path, output_dir):
    """Load one image and preserve the existing diagnostic files and summary."""
    if not image_path.is_file():
        print(f"Error: input image file not found: {image_path}")
        return
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: could not read input image: {image_path}")
        return

    _, metadata, diagnostic_images = process_frame(image, diagnostics=True, verbose=True)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"Error: could not create output directory: {error}")
        return

    saves_succeeded = True
    for suffix, output_image in diagnostic_images.items():
        output_path = output_dir / f"{image_path.stem}_{suffix}.jpg"
        try:
            saved = cv2.imwrite(str(output_path), output_image)
        except (OSError, cv2.error) as error:
            print(f"Error: could not save {output_path}: {error}")
            saves_succeeded = False
            continue
        if saved:
            print(f"Image saved to {output_path}")
        else:
            print(f"Error: could not save image: {output_path}")
            saves_succeeded = False

    # Image CLI success continues to mean processing and saving completed.
    metadata["success"] = saves_succeeded
    return metadata


def main():
    # Select one filename or all supported images in test_images.
    parser = argparse.ArgumentParser(description="Experiment with lane detection on test images.")
    parser.add_argument("image_filename", nargs="?", default=None)
    parser.add_argument("--all", action="store_true", help="Process all JPG, JPEG, and PNG images")
    args = parser.parse_args()
    if args.all and args.image_filename is not None:
        parser.error("choose either an image filename or --all")

    project_root = Path(__file__).resolve().parent.parent
    test_dir = project_root / "test_images"
    output_dir = project_root / "output_images"
    if args.all:
        try:
            image_paths = sorted(
                path for path in test_dir.iterdir()
                if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
        except OSError as error:
            print(f"Error: could not list test images: {error}")
            print("Successfully processed images: 0; failures: 1")
            return
        if not image_paths:
            print("No supported images found in test_images.")
    else:
        filename = args.image_filename or "solidYellowCurve.jpg"
        if Path(filename).name != filename:
            print("Error: provide only an image filename from test_images.")
            print("Successfully processed images: 0; failures: 1")
            return
        image_paths = [test_dir / filename]

    successes = 0
    failures = 0
    for image_path in image_paths:
        print(f"\nProcessing {image_path.name}")
        # Isolate failures so the remaining batch can still be processed.
        try:
            result = process_image(image_path, output_dir)
        except Exception as error:
            print(f"Error processing {image_path.name}: {error}")
            result = None

        if result is None:
            failures += 1
            print(
                f"Summary {image_path.name}: processing failed; fits and curvature "
                "unavailable; offset unavailable; validation unavailable; overlay not generated."
            )
            continue

        radius_labels = {}
        for side in ("Left", "Right"):
            radius = result["radii"].get(side)
            radius_labels[side] = "unavailable" if radius is None else f"{radius:.2f} pixels"
        offset = result["vehicle_offset"]
        offset_label = "unavailable"
        if offset is not None:
            offset_label = (
                f"{offset['pixels']:+.2f} pixels (+right, -left) at y={offset['y']}, "
                f"lane center={offset['lane_center']:.2f}, vehicle center={offset['vehicle_center']:.2f} pixels"
            )
        print(
            f"Summary {image_path.name}: left fit={result['left_fit']}, "
            f"right fit={result['right_fit']}; "
            f"left radius={radius_labels['Left']}, right radius={radius_labels['Right']}; "
            f"vehicle offset={offset_label}; "
            f"validation={result['validation']} ({result['validation_reason']}); "
            f"overlay generated={result['overlay']}; saves successful={result['success']}"
        )
        if result["success"]:
            successes += 1
        else:
            failures += 1

    print(f"\nSuccessfully processed images: {successes}; failures: {failures}")


if __name__ == "__main__":
    main()
