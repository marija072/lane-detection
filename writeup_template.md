## Writeup Template

### You use this file as a template for your writeup.

---

**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

<!-- Image references -->

[image1]: docs/calibration10_comparison.jpg "Calibration distortion correction"
[image2]: docs/solidYellowCurve_perspective_source_points.jpg "Perspective source points"
[image3]: docs/solidYellowCurve_hsv_canny_roi.jpg "HSV and Canny ROI mask"
[image4]: docs/solidYellowCurve_hsv_canny_birds_eye.jpg "Bird's-eye binary mask"
[image5]: docs/solidYellowCurve_sliding_windows.jpg "Sliding-window candidates"
[image6]: docs/solidYellowCurve_polynomial_lanes.jpg "Quadratic lane candidates"
[image7]: docs/solidYellowCurve_lane_overlay.jpg "Lane area in the original camera view"

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

This writeup documents the completed image-based pipeline and its development experiments. Road-camera calibration compatibility remains unverified, and video work is still TODO. Figures are existing diagnostic outputs copied from `output_images/` into `docs/`, without rerunning the programs for this update. Earlier intermediate figures do not have recorded run settings; the final overlay was visually checked to contain no text annotations. The terminal examples and adaptive experiment counts below are verified development results supplied for this writeup, not newly executed tests or ground-truth accuracy measurements.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In [src/camera_calibration.py](src/camera_calibration.py), `main()` searches `camera_cal/` for supported image files and uses a chessboard with **9 columns and 6 rows of internal corners**. It constructs a planar 54-point object grid with `np.mgrid`, using one chessboard square as the coordinate unit. Images are converted to grayscale with `cv2.cvtColor()`. `cv2.findChessboardCorners()` uses adaptive thresholding and image normalization; `cv2.cornerSubPix()` refines detected corners with an `(11, 11)` search window, at most 30 iterations, and an epsilon of `0.001`. Unreadable images, unsuccessful detections, and images with inconsistent dimensions are skipped.

The collected object/image point pairs are passed to `cv2.calibrateCamera()`. The script prints the RMS reprojection error and saves `camera_matrix` and `distortion_coefficients` to `calibration_data/camera_calibration.npz`. No numerical calibration accuracy is claimed here because a calibration run was not performed for this writeup.

In [src/undistort_image.py](src/undistort_image.py), `main()` loads those arrays with `np.load(..., allow_pickle=False)` and applies `cv2.undistort()` to `camera_cal/calibration10.jpg`. It saves the corrected image and a labeled side-by-side comparison, preserving image dimensions without cropping or resizing.

![Original calibration image on the left and undistorted image on the right][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The calibration comparison above provides an existing distortion-correction example. The road-image pipeline in [src/color_threshold.py](src/color_threshold.py), `process_image()`, currently processes the original BGR input; it does **not** apply camera calibration before thresholding or warping.

The earlier straight-line experiment in [src/lane_detection.py](src/lane_detection.py), `main()`, has a `USE_CAMERA_CALIBRATION` option. When enabled, it loads the saved calibration arrays, validates them, and calls `cv2.undistort()` before grayscale processing. Invalid calibration stops that image's processing, and corrected outputs receive a `_calibrated` suffix. The calibration camera may differ from the road-image camera, so this is an experiment rather than evidence of improved road geometry.

**TODO: Complete the road-image distortion-correction example using verified calibration for the road camera.** Existing calibrated line-detection outputs are not a standalone corrected road image and do not establish camera compatibility.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The current binary-mask pipeline is implemented in `process_image()` in [src/color_threshold.py](src/color_threshold.py). `main()` accepts an optional filename from `test_images/` (default `solidYellowCurve.jpg`) or `--all` for `.jpg`, `.jpeg`, and `.png` files. Outputs use each image's stem, and batch processing reports failures without stopping the remaining images.

`cv2.cvtColor(image, cv2.COLOR_BGR2HSV)` converts the original image to HSV. `cv2.inRange()` selects yellow pixels between `[15, 80, 80]` and `[35, 255, 255]`, and white pixels between `[0, 0, 200]` and `[179, 50, 255]`. These are OpenCV HSV values (hue 0–179, saturation/value 0–255). The masks are combined with `cv2.bitwise_or()`. These fixed thresholds may need adjustment under different lighting conditions.

Separately, the original image is converted to grayscale, blurred with `cv2.GaussianBlur(gray, (5, 5), 0)`, and processed with `cv2.Canny(blurred, 100, 170)`. The combined HSV mask and Canny edges are joined with `cv2.bitwise_or()`, retaining either color or edge evidence. This can retain unrelated objects and road edges as well as lane markings.

The ROI is created with `np.zeros_like()` and `cv2.fillPoly()`, then applied using `cv2.bitwise_and()`. Its vertices, expressed as fractions of width and height, are `(0.35, 0.55)`, `(0.65, 0.55)`, `(0.95, 0.95)`, and `(0.00, 0.95)`. The wide lower portion accommodates curved markings. Separate color masks, the color-filtered image, ROI mask/overlay, Canny image, and combined ROI mask are retained as diagnostics. Masks are binary in memory, although JPEG diagnostic files may introduce compression artifacts.

![Existing combined HSV and Canny mask within the road ROI][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In `process_image()` in [src/color_threshold.py](src/color_threshold.py), `cv2.getPerspectiveTransform(source_points, destination_points)` computes a homography, and `cv2.warpPerspective()` applies it to the combined HSV/Canny ROI mask. Output dimensions remain `(width, height)`. `cv2.INTER_NEAREST` avoids interpolating intermediate values into the in-memory binary mask.

The fixed source and destination points are ordered top-left, top-right, bottom-right, bottom-left. Each pair below is a fraction of image width and height:

| Corner | Fixed source | Destination |
| --- | --- | --- |
| Top-left | `(0.45, 0.60)` | `(0.25, 0.05)` |
| Top-right | `(0.55, 0.60)` | `(0.75, 0.05)` |
| Bottom-right | `(0.82, 0.95)` | `(0.75, 0.95)` |
| Bottom-left | `(0.20, 0.95)` | `(0.25, 0.95)` |

The independent experiment in [src/perspective_transform.py](src/perspective_transform.py), `main()`, uses the same fixed point definitions to warp the color road image with `cv2.INTER_LINEAR`. Its source trapezoid is outlined before saving. The main pipeline also saves a labeled source-point visualization.

![Existing source-point diagnostic showing the road trapezoid][image2]

![Existing warped HSV and Canny binary mask][image4]

The fixed transform is a manually selected planar-road approximation. It does not adapt to camera pose, hills, or different lane geometry, and it does not guarantee parallel boundaries or constant pixel lane width. It establishes no pixel-to-meter scale. The example retains unrelated features, so a bird's-eye image alone is not proof of correct lane detection.

The **experimental** adaptive option is retained, but the current default is `USE_ADAPTIVE_PERSPECTIVE = False`. `estimate_source_points(mask)` uses `cv2.HoughLinesP()` on the HSV/Canny ROI mask with rho `1`, theta `np.pi / 180`, threshold `50`, minimum line length `40`, and maximum gap `10`. It skips vertical segments and slopes with magnitude below `0.3`, groups negative/positive slopes, and estimates positions at `0.60 * height` and `0.95 * height`. Each side needs at least three segments and vertical support within `0.05 * height` of those heights. Median positions are used; the 75th percentile of deviations must not exceed `0.08 * width`. Finite, in-image coordinates and a widening, correctly ordered trapezoid are required: top width is `0.05–0.40 * width`, bottom width is `0.25–0.95 * width`, and bottom width must exceed `1.2` times top width. Failed checks select the fixed fallback, and the selected mode is printed. The selected points are shared by the forward warp, source-point diagnostic, and inverse overlay.

In the development batch with adaptive selection enabled, **17 images were processed: only `solidYellowLeft.jpg` used the adaptive transform; the other 16 used the fixed fallback**. The default was subsequently set to `False` because acceptance was low and no visual improvement had been demonstrated. This was a practical experiment decision, not a proven accuracy comparison between fixed and adaptive transforms. Straight-segment estimates can be unreliable on curved or unrelated markings. The earlier saved intermediate figures do not record which transform mode was selected.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane-pixel search and quadratic fitting are also in `process_image()` in [src/color_threshold.py](src/color_threshold.py). A column histogram is calculated with `np.sum(birds_eye_mask[height // 2:, :], axis=0)`, using the bottom half of the warped mask. `np.argmax()` locates one initial candidate in each horizontal half; an empty or all-zero half provides no starting candidate. The histogram graph and candidate-position image are saved for inspection.

The sliding-window search uses **9 horizontal windows**, an **80-pixel horizontal margin**, and a **30-pixel recentering minimum**. It moves from bottom to top, collects nonzero mask pixels within each window, and shifts the next window to their mean x-coordinate when at least 30 pixels are found. Otherwise it retains the previous center. Left and right pixel coordinates are collected separately. These are candidate pixels, not confirmed lane labels; unrelated markings and overlapping search windows can contaminate them.

![Existing sliding-window search, with left windows green and right windows blue][image5]

For each side, non-finite points are removed. At least three points at three distinct y-coordinates are required. `np.polyfit(y_points, x_points, 2, full=True)` fits `x(y) = a*y**2 + b*y + c`; fits with insufficient rank or non-finite coefficients are skipped. `np.polyval()` evaluates valid fits, with coefficients printed separately for each side. Diagnostic left curves are green and right curves blue. The current code draws only between each side's minimum and maximum observed candidate y-values and breaks drawing outside the image.

![Existing quadratic-fit diagnostic; saved results may predate the current supported-range drawing restriction][image6]

The current code also checks fitted geometry before allowing the final polygon: non-finite coordinates and non-positive separation reject the pair. Excessive width variation is rejected only when the maximum exceeds four times the minimum **and** the width spread exceeds half the image width, allowing moderate variation. The polygon is restricted to the overlap of the two observed y-ranges with no extension; it requires at least two rows and 10% of image height. Invalid/out-of-image portions are excluded and disconnected valid sections remain separate. These checks reduce implausible overlays but do not establish that the selected pixels belong to the correct lane.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In [src/color_threshold.py](src/color_threshold.py), `process_image()` reuses the separate left and right quadratic fits described in step 4: `x(y) = a*y**2 + b*y + c`. `radius_of_curvature(coefficients, y)` calculates each boundary's radius using the first derivative `2*a*y + b` and second derivative `2*a`:

```text
R = (1 + (2*a*y + b)**2)**1.5 / abs(2*a)
```

The existing curvature calculation evaluates each available fit at `y = height - 1`. If `abs(a) < 1e-10`, it reports an infinite radius for an effectively straight fit instead of dividing by zero. Missing fits are reported as unavailable. These values are **radii in warped-image pixels, not meters**. Unlike the supported overlay and offset, this curvature evaluation can be below the observed pixel range, so it can describe an extrapolated fit and must be interpreted cautiously.

The horizontal vehicle offset is also calculated in `process_image()`. Both fits must exist, pass geometry validation, and have sufficient overlapping vertical support. Within that shared range, the code selects the lowest valid image row (the largest y) where both fitted x-coordinates are finite, inside the image, and ordered left before right. It does not extrapolate below the supported range for this calculation. If these conditions cannot be met, the offset is unavailable.

The calculation assumes that the camera, and therefore the estimated vehicle center, is horizontally centered at `image_width / 2` in the bird's-eye image:

```text
lane_center_x = (left_x + right_x) / 2
vehicle_center_x = image_width / 2
offset_px = vehicle_center_x - lane_center_x
```

A **positive** offset places the estimated vehicle to the **right** of lane center; a **negative** offset places it to the **left**. The terminal prints the evaluation y-coordinate, lane center, vehicle center, and signed pixel offset, and the per-image summary includes the available measurements.

Verified development terminal examples are:

| Input image | Signed vehicle offset |
| --- | ---: |
| `straight_lines2.jpg` | -0.99 px |
| `solidYellowCurve.jpg` | -15.88 px |
| `challange00101.jpg` | -29.94 px |

All three negative values indicate an estimated vehicle position left of lane center under this assumption. These examples demonstrate reported pixel offsets, not verified physical displacement. No justified pixel-to-meter scale has been established, and camera placement and road-camera calibration compatibility remain unverified. Neither offset nor curvature is converted to meters or based on an assumed physical lane width.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The final visualization is produced by `process_image()` in [src/color_threshold.py](src/color_threshold.py). An empty BGR image is created in bird's-eye coordinates. The two polynomials are evaluated only for polygon construction within their shared observed y-range. Rows must have finite, in-image coordinates with the left boundary before the right; disconnected valid sections are handled separately. Each safe polygon is filled green with `cv2.fillPoly()`, with the left boundary drawn red and the right boundary blue. Geometry validation and the minimum shared-support requirement from step 4 must pass before the overlay is attempted. Missing fits, rejected geometry, insufficient support, or degenerate polygons produce a clear skip message.

`cv2.getPerspectiveTransform(destination_points, source_points)` computes the inverse homography using the same selected points as the forward transform. `cv2.warpPerspective()` maps the colored lane area back to the original image dimensions. `cv2.addWeighted(image, 1.0, warped_lane_area, 0.4, 0)` blends the visualization with the original road image so the road remains visible. The output is `<image_stem>_lane_overlay.jpg`.

![Detected lane area overlaid on solidYellowCurve.jpg, without text annotations][image7]

This example was copied from the current `output_images/solidYellowCurve_lane_overlay.jpg` and visually checked to contain no text annotations. The final image shows only the lane-area and boundary visualization; curvature, offset, support, and validation messages remain in the terminal. The pictured result illustrates lane-area projection on this image, not a quantitative accuracy result or a claim of robustness across the test set.

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

**TODO: Complete the video pipeline and provide a reviewed output video.** The current implementation processes individual images or batches of images; it does not implement temporal tracking or a validated video pipeline.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Development began with the straight-segment experiment in [src/lane_detection.py](src/lane_detection.py): grayscale conversion, a `(5, 5)` Gaussian blur, Canny thresholds `100/170`, and probabilistic Hough detection. Full-image and ROI results were saved separately, followed by slope-filtered left/right segments and straight fits. Running the same straight-line approach on `solidYellowCurve.jpg` motivated a separate curved-lane experiment based on color masks, a bird's-eye view, sliding windows, and quadratic fits. This progression was an implementation experiment, not a measured accuracy benchmark.

Visual diagnostics guided coordinate changes. The color ROI's lower-left edge initially clipped the yellow marking and was widened to `0.00 * width`, while retaining room for the right dashed marking. Perspective experiments narrowed an overly broad source trapezoid and later moved its lower-right x-coordinate from `0.72` to `0.82`. A temporary `straight_lines1.jpg` top-right override of `0.53` was removed; the shared top-right fraction is `0.55` for every image. These were manual alignment decisions, not evidence that one homography is correct for every camera or road.

The adaptive-perspective batch supplied a concrete result: 1 of 17 images (`solidYellowLeft.jpg`) accepted the adaptive estimate, while 16 used fixed fallback. With low acceptance and no demonstrated visual improvement, fixed perspective became the default. The experimental estimator remains available. This result measures acceptance under its checks, not lane-detection accuracy, and does not prove that fixed perspective is more accurate.

Later changes limited unsupported visualization: diagnostic curves stop at their observed candidate y-ranges, and final polygons require sufficient overlap without extending below it. Geometry checks reject crossed boundaries and extreme width inconsistency while allowing moderate variation. The offset examples in step 5 and the text-free overlay in step 6 document current image-based outputs. They do not verify the selected pixels against labeled lanes, establish real-world distances, or validate all images in the batch. Curvature still evaluates at the image bottom even when that row is outside observed support; this is a remaining interpretation limitation.

Observed intermediate masks contain unrelated road edges and bright features, and fixed perspective does not guarantee constant lane width. Additional expected failure cases include shadows, glare, faded paint, occlusion, hills, camera-pose changes, and neighboring markings that attract a histogram peak or sliding window. Slope filtering can confuse boundaries on strong curves, and overlapping windows can collect the same pixels. Geometry checks can reject implausible pairs but cannot establish lane identity. Calibration-camera compatibility is still unverified, so physical curvature and offset claims would be unjustified.

**Proposed future work, not tested improvements:** review predictions against labeled lane boundaries; log reproducible per-image configurations and rejection reasons; verify road-camera calibration and a defensible geometric scale; evaluate lighting-robust thresholds and stronger candidate selection; assess support-aware curvature evaluation and fitting confidence. Video work could investigate temporal tracking, smoothing, and recovery after lost detections, but their effects have not been tested here.

**TODO after video testing:** extend this discussion with observed temporal failures, reviewed video examples, and evidence for any proposed robustness improvements. No video-performance results are claimed at this checkpoint.

