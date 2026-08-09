"""Analyze player poses and positions in a table-tennis video."""

import csv
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from ultralytics import YOLO

matplotlib.use("Agg")
from matplotlib import pyplot as plt


# The small model was more consistent than nano on this video, at about 2x the runtime.
MODEL_NAME = "yolo26s-pose.pt"
INPUT_VIDEO_PATH = Path("input.mp4")
OUTPUT_VIDEO_PATH = Path("output_with_detections.mp4")
POSITIONS_CSV_PATH = Path("player_positions.csv")
HEATMAP_PATH = Path("player_position_heatmap.png")

# END_FRAME is inclusive. Set END_FRAME to 0 to process the entire video.
START_FRAME = 0
END_FRAME = 200
CONFIDENCE = 0.25

Position = tuple[float, float]


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


def find_player_positions(result: Any) -> tuple[Position | None, Position | None]:
    """Return centers of the leftmost and rightmost detected person boxes."""
    boxes = result.boxes.xyxy.cpu().numpy()
    if len(boxes) < 2:
        return None, None

    centers = [
        ((float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2)
        for x1, y1, x2, y2 in boxes
    ]
    centers.sort(key=lambda position: position[0])
    return centers[0], centers[-1]


def csv_position(position: Position | None) -> tuple[str | float, str | float]:
    """Convert a position to rounded CSV values, or blanks when it is missing."""
    if position is None:
        return "", ""
    return round(position[0], 2), round(position[1], 2)


def create_heatmap(
    background_frame: np.ndarray,
    player1_positions: list[Position],
    player2_positions: list[Position],
) -> None:
    """Overlay a combined player-position density map on a video frame."""
    all_positions = player1_positions + player2_positions
    if not all_positions:
        raise RuntimeError("No player positions were found for the heatmap.")

    height, width = background_frame.shape[:2]
    positions = np.asarray(all_positions)
    density, _, _ = np.histogram2d(
        positions[:, 0],
        positions[:, 1],
        bins=(80, 45),
        range=((0, width), (0, height)),
    )

    # histogram2d returns x as the first axis. Transpose it to image row/column order.
    density = cv2.GaussianBlur(density.T, (0, 0), sigmaX=1.5, sigmaY=1.5)
    visible_density = np.ma.masked_where(density <= 0.01, density)

    figure, axis = plt.subplots(figsize=(14, 8))
    axis.imshow(cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB))
    heatmap = axis.imshow(
        visible_density,
        cmap="turbo",
        alpha=0.65,
        interpolation="bilinear",
        extent=(0, width, height, 0),
    )
    axis.set_title("Combined Player Position Heatmap")
    axis.set_xlabel("X Position")
    axis.set_ylabel("Y Position")
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    figure.colorbar(heatmap, ax=axis, label="Density")
    figure.tight_layout()
    figure.savefig(HEATMAP_PATH, dpi=160)
    plt.close(figure)


def analyze_video() -> tuple[int, int, int]:
    """Create the annotated video, position CSV, and player heatmap."""
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
        missing_position_frames = 0
        background_frame: np.ndarray | None = None
        player1_positions: list[Position] = []
        player2_positions: list[Position] = []

        with POSITIONS_CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                [
                    "frame",
                    "player1_position_x",
                    "player1_position_y",
                    "player2_position_x",
                    "player2_position_y",
                ]
            )

            while END_FRAME == 0 or frame_number <= END_FRAME:
                success, frame = capture.read()
                if not success:
                    break

                if background_frame is None:
                    background_frame = frame.copy()

                result = model.predict(
                    source=frame,
                    conf=CONFIDENCE,
                    verbose=False,
                )[0]
                player1, player2 = find_player_positions(result)

                if player1 is None or player2 is None:
                    missing_position_frames += 1
                else:
                    player1_positions.append(player1)
                    player2_positions.append(player2)

                player1_x, player1_y = csv_position(player1)
                player2_x, player2_y = csv_position(player2)
                csv_writer.writerow(
                    [frame_number, player1_x, player1_y, player2_x, player2_y]
                )

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

        if background_frame is None:
            raise RuntimeError("No video frames were processed.")

        create_heatmap(background_frame, player1_positions, player2_positions)
        return written_frames, len(player1_positions), missing_position_frames
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main() -> None:
    """Validate settings, process the video, and report the result."""
    validate_settings()
    written_frames, position_frames, missing_position_frames = analyze_video()

    if written_frames == 0:
        raise RuntimeError("No video frames were processed.")

    print(f"Finished processing {written_frames} frame(s).")
    print(f"Recorded both player positions in {position_frames} frame(s).")
    if missing_position_frames:
        print(f"Missing two-player detections in {missing_position_frames} frame(s).")
    print(f"Saved annotated video to: {OUTPUT_VIDEO_PATH.resolve()}")
    print(f"Saved player positions to: {POSITIONS_CSV_PATH.resolve()}")
    print(f"Saved player heatmap to: {HEATMAP_PATH.resolve()}")


if __name__ == "__main__":
    main()
