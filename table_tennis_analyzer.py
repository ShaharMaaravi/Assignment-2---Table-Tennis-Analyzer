"""Draw person bounding boxes and poses on frames from a table-tennis video."""

from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_NAME = "yolo26n-pose.pt"
INPUT_VIDEO_PATH = Path("input.mp4")
OUTPUT_VIDEO_PATH = Path("output_with_detections.mp4")

# END_FRAME is inclusive. Set END_FRAME to 0 to process the entire video.
START_FRAME = 0
END_FRAME = 200
CONFIDENCE = 0.25


def validate_settings() -> None:
    """Check the input path and frame-range constants."""
    if not INPUT_VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO_PATH}")
    if INPUT_VIDEO_PATH.resolve() == OUTPUT_VIDEO_PATH.resolve():
        raise ValueError("The output path must be different from the input path.")
    if START_FRAME < 0:
        raise ValueError("START_FRAME cannot be negative.")
    if END_FRAME < 0:
        raise ValueError("END_FRAME cannot be negative.")
    if END_FRAME != 0 and END_FRAME < START_FRAME:
        raise ValueError("END_FRAME must be 0 or greater than or equal to START_FRAME.")
    if not 0.0 <= CONFIDENCE <= 1.0:
        raise ValueError("CONFIDENCE must be between 0 and 1.")


def create_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    """Create an MP4 writer and ensure that it opened successfully."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), codec, fps, frame_size)

    if not writer.isOpened():
        raise OSError(f"Could not create output video: {output_path}")

    return writer


def analyze_video() -> int:
    """Run pose detection on the selected frames and return the number written."""
    capture = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
    if not capture.isOpened():
        raise OSError(f"Could not open input video: {INPUT_VIDEO_PATH}")

    writer: cv2.VideoWriter | None = None

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("The input video has invalid dimensions or frame rate.")
        if total_frames > 0 and START_FRAME >= total_frames:
            raise ValueError(
                f"START_FRAME ({START_FRAME}) is outside the video "
                f"({total_frames} frames)."
            )

        writer = create_video_writer(OUTPUT_VIDEO_PATH, fps, (width, height))
        model = YOLO(MODEL_NAME)

        capture.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
        frame_number = START_FRAME
        written_frames = 0

        while END_FRAME == 0 or frame_number <= END_FRAME:
            success, frame = capture.read()
            if not success:
                break

            result = model.predict(
                source=frame,
                conf=CONFIDENCE,
                verbose=False,
            )[0]
            annotated_frame = result.plot(
                boxes=True,
                labels=True,
                kpt_line=True,
                kpt_radius=5,
                line_width=2,
            )
            writer.write(annotated_frame)

            written_frames += 1
            frame_number += 1

            if written_frames % 25 == 0:
                print(f"Processed {written_frames} frame(s)...")

        return written_frames
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main() -> None:
    """Validate settings, process the video, and report the result."""
    validate_settings()
    written_frames = analyze_video()

    if written_frames == 0:
        raise RuntimeError("No video frames were processed.")

    print(f"Finished processing {written_frames} frame(s).")
    print(f"Saved annotated video to: {OUTPUT_VIDEO_PATH.resolve()}")


if __name__ == "__main__":
    main()
