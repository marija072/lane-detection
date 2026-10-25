"""Process a video frame by frame with the existing lane detector."""

import argparse
import math
from pathlib import Path

import cv2

from color_threshold import CalibrationError, load_calibration, process_frame, undistort_frame


def main():
    parser = argparse.ArgumentParser(description="Add lane overlays to a test video.")
    parser.add_argument("video_filename", help="Video filename inside test_videos")
    parser.add_argument("--undistort", action="store_true", help="Apply saved camera calibration")
    args = parser.parse_args()
    if Path(args.video_filename).name != args.video_filename:
        parser.error("provide only a filename from test_videos")

    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "test_videos" / args.video_filename
    suffix = "_undistorted" if args.undistort else ""
    output_path = project_root / "output_videos" / f"{input_path.stem}_lane_overlay{suffix}.mp4"
    if not input_path.is_file():
        print(f"Error: input video not found: {input_path}")
        return 1

    capture = None
    writer = None
    frame_count = 0
    successful_detections = 0
    rejected_detections = 0
    processing_errors = 0
    completed = False
    try:
        calibration = load_calibration(project_root / "calibration_data" / "camera_calibration.npz") if args.undistort else None
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {input_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"invalid input FPS: {fps}; no replacement was assumed")
        expected_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)

        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError("could not read the first video frame")
        height, width = frame.shape[:2]
        if calibration is not None:
            # Check compatibility before opening the output writer.
            _, scale = undistort_frame(frame, calibration)
            print(f"Distortion correction enabled: matrix scale={scale:g}")
        else:
            print("Distortion correction disabled")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open output writer: {output_path}")
        print(f"Processing {input_path.name}: {width}x{height}, {fps:g} FPS")

        while True:
            if frame.shape != (height, width, 3):
                raise RuntimeError(f"frame {frame_count + 1} has unexpected dimensions: {frame.shape}")

            # No diagnostic files, per-frame measurements, or temporal tracking.
            try:
                result_frame, metadata, _ = process_frame(frame, calibration=calibration)
                detected = metadata["success"]
                if result_frame.shape != frame.shape or result_frame.dtype != frame.dtype:
                    raise ValueError("lane detector returned an incompatible frame")
            except CalibrationError:
                raise
            except Exception:
                # Preserve this frame even if detection raises an unexpected error.
                result_frame = undistort_frame(frame, calibration)[0] if calibration is not None else frame
                detected = False
                processing_errors += 1

            if not writer.isOpened():
                raise RuntimeError("output writer closed unexpectedly")
            # OpenCV write() has no reliable success return; exceptions are detected.
            writer.write(result_frame)
            frame_count += 1
            if detected:
                successful_detections += 1
            else:
                rejected_detections += 1

            ok, frame = capture.read()
            if not ok:
                # Some backends report EOF and decoding failures the same way.
                if math.isfinite(expected_frames) and expected_frames > 0:
                    if frame_count < round(expected_frames):
                        raise RuntimeError(
                            f"reading stopped early: {frame_count} of approximately "
                            f"{round(expected_frames)} frames; output is partial"
                        )
                else:
                    print("Read ended; frame count unavailable, so EOF cannot be distinguished from a read failure.")
                completed = True
                break
            if frame is None or frame.size == 0:
                raise RuntimeError(f"empty frame after {frame_count} frames; output is partial")

    except (OSError, RuntimeError, cv2.error) as error:
        print(f"Error: {error}")
    finally:
        # Finalize the MP4 and release the input even when processing stops early.
        try:
            if writer is not None:
                writer.release()
        except cv2.error as error:
            print(f"Error finalizing output: {error}")
            completed = False
        finally:
            if capture is not None:
                capture.release()
        print(
            f"Frames written: {frame_count}; successful detections: {successful_detections}; "
            f"rejected detections: {rejected_detections}; processing errors: {processing_errors}"
        )

    if completed:
        if not output_path.is_file() or output_path.stat().st_size == 0:
            print("Error: output video is missing or empty")
            return 1
        print(f"Video saved to {output_path}")
        return 0
    if frame_count:
        print(f"Partial output may remain at {output_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
