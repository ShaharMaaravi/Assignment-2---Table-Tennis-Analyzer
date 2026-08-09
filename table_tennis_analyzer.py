"""Analyze player poses and positions in a table-tennis video."""

import csv
from pathlib import Path
from typing import Any

import cv2
import easyocr
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
SCORE_CHART_PATH = Path("score_over_time.png")
EASYOCR_MODEL_DIRECTORY = Path(".easyocr")

# END_FRAME is inclusive. Set END_FRAME to 0 to process the entire video.
START_FRAME = 0
END_FRAME = 200
CONFIDENCE = 0.25

# Side-view validation thresholds, expressed relative to the frame dimensions.
MIN_PLAYER_AREA_RATIO = 0.005
MAX_PLAYER_AREA_RATIO = 0.20
MIN_CANDIDATE_TO_LARGEST_RATIO = 0.40
MIN_PLAYER_SIZE_SIMILARITY = 0.25
MIN_HORIZONTAL_SEPARATION_RATIO = 0.25
MAX_VERTICAL_SEPARATION_RATIO = 0.25
LEFT_PLAYER_MAX_X_RATIO = 0.48
RIGHT_PLAYER_MIN_X_RATIO = 0.52
MIN_VISIBLE_KEYPOINTS = 6
KEYPOINT_CONFIDENCE = 0.25

# The broadcast scoreboard is fixed in the lower-left corner of a 16:9 frame.
SCOREBOARD_LEFT_RATIO = 400 / 1920
SCOREBOARD_RIGHT_RATIO = 600 / 1920
SCOREBOARD_TOP_RATIO = 940 / 1080
SCORE_OCR_INTERVAL = 25
SCORE_OCR_SCALE = 3
SCORE_OCR_MIN_CONFIDENCE = 0.35
SCOREBOARD_ROW_SPLIT_RATIO = 0.30
MAX_GAMES_WON = 7
MAX_CURRENT_GAME_POINTS = 30

Position = tuple[float, float]
Score = tuple[int, int, int, int]


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


def is_valid_frame(frame: np.ndarray, result: Any) -> bool:
    """Return whether detections resemble the normal side view of the match."""
    boxes = result.boxes.xyxy.cpu().numpy()
    if len(boxes) < 2:
        return False

    height, width = frame.shape[:2]
    frame_area = width * height
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Ignore people much smaller than the largest detection (usually the referee),
    # then choose the widest-separated pair. This also tolerates duplicate YOLO boxes.
    candidate_indices = np.flatnonzero(
        areas >= areas.max() * MIN_CANDIDATE_TO_LARGEST_RATIO
    )
    if len(candidate_indices) < 2:
        return False

    candidate_centers_x = (
        boxes[candidate_indices, 0] + boxes[candidate_indices, 2]
    ) / 2
    left_candidate = candidate_indices[np.argmin(candidate_centers_x)]
    right_candidate = candidate_indices[np.argmax(candidate_centers_x)]
    if left_candidate == right_candidate:
        return False

    player_indices = np.asarray([left_candidate, right_candidate])
    player_boxes = boxes[player_indices]
    player_areas = areas[player_indices]

    area_ratios = player_areas / frame_area
    if np.any(area_ratios < MIN_PLAYER_AREA_RATIO):
        return False
    if np.any(area_ratios > MAX_PLAYER_AREA_RATIO):
        return False

    size_similarity = float(player_areas.min() / player_areas.max())
    if size_similarity < MIN_PLAYER_SIZE_SIMILARITY:
        return False

    centers_x = (player_boxes[:, 0] + player_boxes[:, 2]) / 2
    centers_y = (player_boxes[:, 1] + player_boxes[:, 3]) / 2
    centers_x.sort()

    # In the valid side view, the players straddle the table and are at similar heights.
    if centers_x[0] > width * LEFT_PLAYER_MAX_X_RATIO:
        return False
    if centers_x[1] < width * RIGHT_PLAYER_MIN_X_RATIO:
        return False
    if centers_x[1] - centers_x[0] < width * MIN_HORIZONTAL_SEPARATION_RATIO:
        return False
    if abs(float(centers_y[1] - centers_y[0])) > height * MAX_VERTICAL_SEPARATION_RATIO:
        return False

    keypoint_confidences = result.keypoints.conf
    if keypoint_confidences is not None:
        player_keypoints = keypoint_confidences[player_indices].cpu().numpy()
        visible_counts = (player_keypoints >= KEYPOINT_CONFIDENCE).sum(axis=1)
        if np.any(visible_counts < MIN_VISIBLE_KEYPOINTS):
            return False

    return True


def csv_position(position: Position | None) -> tuple[str | float, str | float]:
    """Convert a position to rounded CSV values, or blanks when it is missing."""
    if position is None:
        return "", ""
    return round(position[0], 2), round(position[1], 2)


def read_score(frame: np.ndarray, reader: easyocr.Reader) -> Score | None:
    """Read the four scoreboard digits as games/points for the two players."""
    height, width = frame.shape[:2]
    x1 = int(width * SCOREBOARD_LEFT_RATIO)
    x2 = int(width * SCOREBOARD_RIGHT_RATIO)
    y1 = int(height * SCOREBOARD_TOP_RATIO)
    scoreboard = frame[y1:height, x1:x2]

    if scoreboard.size == 0:
        return None

    scoreboard = cv2.resize(
        scoreboard,
        None,
        fx=SCORE_OCR_SCALE,
        fy=SCORE_OCR_SCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    detections = reader.readtext(
        scoreboard,
        allowlist="0123456789",
        detail=1,
        paragraph=False,
    )

    # Assign each OCR result to one of the fixed 2x2 scoreboard cells.
    cells: dict[tuple[int, int], tuple[float, int]] = {}
    score_height, score_width = scoreboard.shape[:2]

    for box, text, confidence in detections:
        digits = "".join(character for character in text if character.isdigit())
        confidence = float(confidence)
        if not digits or confidence < SCORE_OCR_MIN_CONFIDENCE:
            continue

        center_x = sum(float(point[0]) for point in box) / 4
        center_y = sum(float(point[1]) for point in box) / 4
        row = 0 if center_y < score_height * SCOREBOARD_ROW_SPLIT_RATIO else 1
        column = 0 if center_x < score_width / 2 else 1
        value = int(digits)

        maximum = MAX_GAMES_WON if column == 0 else MAX_CURRENT_GAME_POINTS
        if value > maximum:
            continue

        existing = cells.get((row, column))
        if existing is None or confidence > existing[0]:
            cells[(row, column)] = (confidence, value)

    required_cells = ((0, 0), (0, 1), (1, 0), (1, 1))
    if not all(cell in cells for cell in required_cells):
        return None

    return tuple(cells[cell][1] for cell in required_cells)


def calculated_scores(score: Score) -> tuple[int, int]:
    """Convert games won and current points to one chart value per player."""
    player1_games, player1_points, player2_games, player2_points = score
    return (
        player1_games * 10 + player1_points,
        player2_games * 10 + player2_points,
    )


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


def create_score_chart(
    frame_numbers: list[int],
    player1_scores: list[int],
    player2_scores: list[int],
) -> None:
    """Save both players' calculated scores as step lines over source frames."""
    figure, axis = plt.subplots(figsize=(12, 6))

    if frame_numbers:
        axis.step(
            frame_numbers,
            player1_scores,
            where="post",
            label="Player 1 (top scoreboard row)",
            linewidth=2,
        )
        axis.step(
            frame_numbers,
            player2_scores,
            where="post",
            label="Player 2 (bottom scoreboard row)",
            linewidth=2,
        )
        axis.legend()
    else:
        axis.text(
            0.5,
            0.5,
            "No scoreboard was detected in the selected frame range",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    axis.set_title("Player Scores Over Time")
    axis.set_xlabel("Source Frame")
    axis.set_ylabel("Calculated Score (games won × 10 + current points)")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(SCORE_CHART_PATH, dpi=160)
    plt.close(figure)


def analyze_video() -> tuple[int, int, int, int, int, int]:
    """Create the annotated video, position CSV, heatmap, and score chart."""
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
        score_reader = easyocr.Reader(
            ["en"],
            gpu=False,
            model_storage_directory=str(EASYOCR_MODEL_DIRECTORY),
            verbose=False,
        )

        capture.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
        frame_number = START_FRAME
        inspected_frames = 0
        written_frames = 0
        invalid_frames = 0
        missing_position_frames = 0
        background_frame: np.ndarray | None = None
        player1_positions: list[Position] = []
        player2_positions: list[Position] = []
        last_score: Score | None = None
        recognized_score_samples = 0
        score_frame_numbers: list[int] = []
        player1_scores: list[int] = []
        player2_scores: list[int] = []

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

                current_frame_number = frame_number
                frame_number += 1
                inspected_frames += 1
                result = model.predict(
                    source=frame,
                    conf=CONFIDENCE,
                    verbose=False,
                )[0]

                if (
                    current_frame_number - START_FRAME
                ) % SCORE_OCR_INTERVAL == 0:
                    recognized_score = read_score(frame, score_reader)
                    if recognized_score is not None:
                        last_score = recognized_score
                        recognized_score_samples += 1

                if not is_valid_frame(frame, result):
                    invalid_frames += 1
                    if inspected_frames % 25 == 0:
                        print(
                            f"Inspected {inspected_frames} frame(s); "
                            f"kept {written_frames}."
                        )
                    continue

                if background_frame is None:
                    background_frame = frame.copy()

                player1, player2 = find_player_positions(result)

                if player1 is None or player2 is None:
                    missing_position_frames += 1
                else:
                    player1_positions.append(player1)
                    player2_positions.append(player2)

                player1_x, player1_y = csv_position(player1)
                player2_x, player2_y = csv_position(player2)
                csv_writer.writerow(
                    [
                        current_frame_number,
                        player1_x,
                        player1_y,
                        player2_x,
                        player2_y,
                    ]
                )

                if last_score is not None:
                    player1_score, player2_score = calculated_scores(last_score)
                    score_frame_numbers.append(current_frame_number)
                    player1_scores.append(player1_score)
                    player2_scores.append(player2_score)

                annotated_frame = result.plot(
                    boxes=True,
                    labels=True,
                    kpt_line=True,
                    kpt_radius=5,
                    line_width=2,
                )
                writer.write(annotated_frame)

                written_frames += 1

                if inspected_frames % 25 == 0:
                    print(
                        f"Inspected {inspected_frames} frame(s); "
                        f"kept {written_frames}."
                    )

        if background_frame is None:
            raise RuntimeError("No video frames were processed.")

        create_heatmap(background_frame, player1_positions, player2_positions)
        create_score_chart(score_frame_numbers, player1_scores, player2_scores)
        return (
            inspected_frames,
            written_frames,
            invalid_frames,
            len(player1_positions),
            missing_position_frames,
            recognized_score_samples,
        )
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main() -> None:
    """Validate settings, process the video, and report the result."""
    validate_settings()
    (
        inspected_frames,
        written_frames,
        invalid_frames,
        position_frames,
        missing_position_frames,
        recognized_score_samples,
    ) = analyze_video()

    if written_frames == 0:
        raise RuntimeError("No video frames were processed.")

    print(f"Finished inspecting {inspected_frames} frame(s).")
    print(f"Kept {written_frames} valid frame(s); skipped {invalid_frames} invalid frame(s).")
    print(f"Recorded both player positions in {position_frames} frame(s).")
    if missing_position_frames:
        print(f"Missing two-player detections in {missing_position_frames} frame(s).")
    print(f"Recognized the scoreboard in {recognized_score_samples} OCR sample(s).")
    print(f"Saved annotated video to: {OUTPUT_VIDEO_PATH.resolve()}")
    print(f"Saved player positions to: {POSITIONS_CSV_PATH.resolve()}")
    print(f"Saved player heatmap to: {HEATMAP_PATH.resolve()}")
    print(f"Saved score chart to: {SCORE_CHART_PATH.resolve()}")


if __name__ == "__main__":
    main()
