"""Analyze player poses and positions in a table-tennis video."""

import csv
import json
import os
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import easyocr
import matplotlib
import numpy as np
from dotenv import load_dotenv
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
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
BALL_POSITIONS_CSV_PATH = Path("ball_positions.csv")
BALL_SPEED_CHART_PATH = Path("ball_speed_over_time.png")
BALL_PREDICTION_CACHE_PATH = Path(".roboflow_prediction_cache.json")
RALLY_STATISTICS_CSV_PATH = Path("rally_statistics.csv")
RALLY_ANALYSIS_CHART_PATH = Path("rally_analysis.png")
HIGHLIGHTS_VIDEO_PATH = Path("match_highlights.mp4")
MOMENTUM_CSV_PATH = Path("match_momentum.csv")
MOMENTUM_CHART_PATH = Path("match_momentum.png")
DASHBOARD_HTML_PATH = Path("dashboard.html")
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

# Roboflow ball detector and trajectory settings.
ROBOFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_API_KEY_ENV = "ROBOFLOW_API_KEY"
ROBOFLOW_MODEL_ID_ENV = "ROBOFLOW_MODEL_ID"
BALL_CONFIDENCE = 0.20
BALL_MAX_TRACKING_GAP_FRAMES = 10
BALL_MAX_DISPLACEMENT_PX_PER_FRAME = 120.0
BALL_MAX_SPEED_PX_PER_SECOND = 6000.0
BALL_MIN_MOTION_SCORE = 2.5
BALL_MIN_X_RATIO = 0.12
BALL_MAX_X_RATIO = 0.88
BALL_MIN_Y_RATIO = 0.34
BALL_MAX_Y_RATIO = 0.72
BALL_SPEED_SMOOTHING_WINDOW = 5
BALL_TRAIL_LENGTH = 18
BALL_FAST_SPEED_PX_PER_SECOND = 3000.0

# Smart rally analysis and highlight settings.
RALLY_MAX_BALL_GAP_SECONDS = 0.45
RALLY_MIN_DURATION_SECONDS = 0.75
RALLY_MIN_DETECTIONS = 6
RALLY_HIT_MIN_HORIZONTAL_SPEED_PX_PER_FRAME = 3.0
RALLY_HIT_MIN_DISPLACEMENT_PX = 70.0
RALLY_HIT_MIN_GAP_SECONDS = 0.12
RALLY_CLIP_PADDING_SECONDS = 0.40
MAX_HIGHLIGHT_RALLIES = 3
HIGHLIGHT_TITLE_SECONDS = 1.0
MAX_PLAYER_DISPLACEMENT_PX_PER_FRAME = 45.0

# Match momentum is an explainable heuristic, not a win-probability model.
MOMENTUM_ATTACK_WEIGHT = 22.0
MOMENTUM_MOVEMENT_WEIGHT = 8.0
MOMENTUM_SCORE_WEIGHT = 20.0
MOMENTUM_SMOOTHING_ALPHA = 0.14

Position = tuple[float, float]
Score = tuple[int, int, int, int]


@dataclass(frozen=True)
class BallDetection:
    """One table-tennis-ball prediction in source-frame coordinates."""

    x: float
    y: float
    width: float
    height: float
    confidence: float


@dataclass(frozen=True)
class BallTrackPoint:
    """A detected ball center and its smoothed pixel speed."""

    frame_number: int
    x: float
    y: float
    speed: float | None


@dataclass(frozen=True)
class PlayerPositionSample:
    """Both left/right player centers in one valid source frame."""

    frame_number: int
    player1: Position
    player2: Position


@dataclass(frozen=True)
class RallyStatistics:
    """Summary metrics and estimated hit events for one detected rally."""

    rally_id: int
    start_frame: int
    end_frame: int
    duration_seconds: float
    ball_detections: int
    estimated_hits: int
    hit_frames: tuple[int, ...]
    average_ball_speed: float
    maximum_ball_speed: float
    player1_distance: float
    player2_distance: float
    excitement_score: float


@dataclass(frozen=True)
class MomentumSample:
    """Explainable momentum components for one valid source frame."""

    frame_number: int
    player1_momentum: float
    player2_momentum: float
    score_advantage: float
    attack_signal: float
    movement_efficiency: float
    rally_intensity: float
    rally_id: int | None


def validate_settings() -> None:
    """Check the input path and frame-range constants."""
    load_dotenv()
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
    if not os.getenv(ROBOFLOW_API_KEY_ENV):
        raise EnvironmentError(
            f"Missing {ROBOFLOW_API_KEY_ENV}. Add it to the local .env file."
        )
    if not os.getenv(ROBOFLOW_MODEL_ID_ENV):
        raise EnvironmentError(
            f"Missing {ROBOFLOW_MODEL_ID_ENV}. Add it to the local .env file."
        )


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


def create_roboflow_client() -> InferenceHTTPClient:
    """Create a configured client for the trained Roboflow ball model."""
    api_key = os.environ[ROBOFLOW_API_KEY_ENV]
    client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=api_key)
    client.configure(InferenceConfiguration(confidence_threshold=BALL_CONFIDENCE))
    return client


def find_ball(
    frame: np.ndarray,
    client: InferenceHTTPClient,
    model_id: str,
    previous_frame: np.ndarray | None = None,
    previous_point: BallTrackPoint | None = None,
    frame_number: int | None = None,
    prediction_cache: dict[str, list[dict[str, Any]]] | None = None,
    cache_key: str | None = None,
) -> BallDetection | None:
    """Return the most plausible ball using confidence, motion, and continuity."""
    if prediction_cache is not None and cache_key in prediction_cache:
        response: Any = {"predictions": prediction_cache[cache_key]}
    else:
        response = client.infer(frame, model_id=model_id)
    if isinstance(response, list):
        response = response[0] if response else {}
    if not isinstance(response, dict):
        return None

    predictions = response.get("predictions", [])
    if prediction_cache is not None and cache_key is not None:
        prediction_cache[cache_key] = [
            prediction for prediction in predictions if isinstance(prediction, dict)
        ]
    if previous_frame is None:
        # Motion is required to reject static logos, so do not seed a track from
        # the first source frame when no comparison frame is available.
        return None
    height, width = frame.shape[:2]
    valid_predictions: list[tuple[BallDetection, float]] = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        required_values = ("x", "y", "width", "height", "confidence")
        if not all(key in prediction for key in required_values):
            continue
        if float(prediction["confidence"]) < BALL_CONFIDENCE:
            continue

        detection = BallDetection(
            x=float(prediction["x"]),
            y=float(prediction["y"]),
            width=float(prediction["width"]),
            height=float(prediction["height"]),
            confidence=float(prediction["confidence"]),
        )
        if not width * BALL_MIN_X_RATIO <= detection.x <= width * BALL_MAX_X_RATIO:
            continue
        if not height * BALL_MIN_Y_RATIO <= detection.y <= height * BALL_MAX_Y_RATIO:
            continue

        motion_score = detection_motion_score(frame, previous_frame, detection)
        if previous_frame is not None and motion_score < BALL_MIN_MOTION_SCORE:
            continue
        valid_predictions.append((detection, motion_score))

    if not valid_predictions:
        return None

    if previous_point is not None and frame_number is not None:
        frame_gap = frame_number - previous_point.frame_number
        if 0 < frame_gap <= BALL_MAX_TRACKING_GAP_FRAMES:
            maximum_distance = BALL_MAX_DISPLACEMENT_PX_PER_FRAME * frame_gap
            nearby_predictions = [
                (detection, motion_score)
                for detection, motion_score in valid_predictions
                if np.hypot(
                    detection.x - previous_point.x,
                    detection.y - previous_point.y,
                )
                <= maximum_distance
            ]
            if not nearby_predictions:
                return None
            valid_predictions = nearby_predictions

    def prediction_score(item: tuple[BallDetection, float]) -> float:
        detection, motion_score = item
        motion_bonus = min(motion_score / 30.0, 1.0) * 0.35
        return detection.confidence + motion_bonus

    return max(valid_predictions, key=prediction_score)[0]


def load_ball_prediction_cache() -> dict[str, list[dict[str, Any]]]:
    """Load non-secret raw model predictions saved from earlier API calls."""
    if not BALL_PREDICTION_CACHE_PATH.is_file():
        return {}
    try:
        data = json.loads(BALL_PREDICTION_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_ball_prediction_cache(
    prediction_cache: dict[str, list[dict[str, Any]]],
) -> None:
    """Save API predictions so rerunning analysis does not consume more requests."""
    temporary_path = BALL_PREDICTION_CACHE_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(prediction_cache, separators=(",", ":")), encoding="utf-8"
    )
    temporary_path.replace(BALL_PREDICTION_CACHE_PATH)


def detection_motion_score(
    frame: np.ndarray,
    previous_frame: np.ndarray | None,
    detection: BallDetection,
) -> float:
    """Measure local frame-to-frame change around a predicted tiny ball."""
    if previous_frame is None or previous_frame.shape != frame.shape:
        return 0.0

    height, width = frame.shape[:2]
    radius = max(12, round(max(detection.width, detection.height) * 1.5))
    x1 = max(0, round(detection.x) - radius)
    y1 = max(0, round(detection.y) - radius)
    x2 = min(width, round(detection.x) + radius + 1)
    y2 = min(height, round(detection.y) + radius + 1)
    if x1 >= x2 or y1 >= y2:
        return 0.0

    current_patch = frame[y1:y2, x1:x2]
    previous_patch = previous_frame[y1:y2, x1:x2]
    difference = cv2.absdiff(current_patch, previous_patch)
    return float(cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY).mean())


def calculate_ball_speed(
    detection: BallDetection,
    frame_number: int,
    previous_point: BallTrackPoint | None,
    fps: float,
) -> float | None:
    """Calculate ball speed in pixels/second between consecutive detections."""
    if previous_point is None:
        return None

    frame_gap = frame_number - previous_point.frame_number
    if frame_gap <= 0 or frame_gap > BALL_MAX_TRACKING_GAP_FRAMES:
        return None

    pixel_distance = float(
        np.hypot(detection.x - previous_point.x, detection.y - previous_point.y)
    )
    speed = pixel_distance * fps / frame_gap
    if speed > BALL_MAX_SPEED_PX_PER_SECOND:
        return None
    return speed


def speed_color(speed: float | None) -> tuple[int, int, int]:
    """Map slow-to-fast movement to blue-to-red BGR colors."""
    if speed is None:
        return 0, 255, 255
    ratio = min(max(speed / BALL_FAST_SPEED_PX_PER_SECOND, 0.0), 1.0)
    color_index = np.asarray([[round(ratio * 255)]], dtype=np.uint8)
    color = cv2.applyColorMap(color_index, cv2.COLORMAP_TURBO)[0, 0]
    return int(color[0]), int(color[1]), int(color[2])


def draw_ball_tracking(
    frame: np.ndarray,
    detection: BallDetection | None,
    trail: deque[BallTrackPoint],
) -> None:
    """Draw the current ball box, speed, and recent speed-colored trajectory."""
    trail_points = list(trail)
    for start, end in zip(trail_points, trail_points[1:]):
        if end.frame_number - start.frame_number > BALL_MAX_TRACKING_GAP_FRAMES:
            continue
        cv2.line(
            frame,
            (round(start.x), round(start.y)),
            (round(end.x), round(end.y)),
            speed_color(end.speed),
            3,
            cv2.LINE_AA,
        )

    if detection is None or not trail:
        return

    current = trail[-1]
    x1 = round(detection.x - detection.width / 2)
    y1 = round(detection.y - detection.height / 2)
    x2 = round(detection.x + detection.width / 2)
    y2 = round(detection.y + detection.height / 2)
    color = speed_color(current.speed)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.circle(frame, (round(detection.x), round(detection.y)), 5, color, -1)

    speed_text = "speed: -- px/s"
    if current.speed is not None:
        speed_text = f"speed: {current.speed:.0f} px/s"
    label = f"ball {detection.confidence:.2f} | {speed_text}"
    text_y = max(24, y1 - 10)
    cv2.putText(
        frame,
        label,
        (max(0, x1), text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
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


def create_ball_speed_chart(track_points: list[BallTrackPoint]) -> None:
    """Save the detected ball's smoothed pixel speed over source frames."""
    measured_points = [point for point in track_points if point.speed is not None]
    figure, axis = plt.subplots(figsize=(12, 6))

    if measured_points:
        frame_numbers = [point.frame_number for point in measured_points]
        speeds = [float(point.speed) for point in measured_points]
        axis.plot(frame_numbers, speeds, color="tab:red", linewidth=1.8)
        maximum_index = int(np.argmax(speeds))
        axis.scatter(
            [frame_numbers[maximum_index]],
            [speeds[maximum_index]],
            color="black",
            zorder=3,
            label=f"Fastest: {speeds[maximum_index]:.0f} px/s",
        )
        axis.legend()
    else:
        axis.text(
            0.5,
            0.5,
            "Not enough consecutive ball detections to calculate speed",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    axis.set_title("Table-Tennis Ball Speed Over Time")
    axis.set_xlabel("Source Frame")
    axis.set_ylabel("Smoothed Speed (pixels/second)")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(BALL_SPEED_CHART_PATH, dpi=160)
    plt.close(figure)


def estimate_hit_frames(
    track_points: list[BallTrackPoint], fps: float
) -> tuple[int, ...]:
    """Estimate racket contacts from stable horizontal direction reversals."""
    if len(track_points) < 3:
        return ()

    minimum_hit_gap = max(1, round(fps * RALLY_HIT_MIN_GAP_SECONDS))
    hit_frames: list[int] = []
    previous_direction = 0
    last_turn_x = track_points[0].x
    last_hit_frame = track_points[0].frame_number - minimum_hit_gap

    for previous, current in zip(track_points, track_points[1:]):
        frame_gap = current.frame_number - previous.frame_number
        if frame_gap <= 0 or frame_gap > round(fps * RALLY_MAX_BALL_GAP_SECONDS):
            previous_direction = 0
            last_turn_x = current.x
            continue

        horizontal_speed = (current.x - previous.x) / frame_gap
        if abs(horizontal_speed) < RALLY_HIT_MIN_HORIZONTAL_SPEED_PX_PER_FRAME:
            continue
        direction = 1 if horizontal_speed > 0 else -1

        if (
            previous_direction != 0
            and direction != previous_direction
            and abs(current.x - last_turn_x) >= RALLY_HIT_MIN_DISPLACEMENT_PX
            and current.frame_number - last_hit_frame >= minimum_hit_gap
        ):
            hit_frames.append(current.frame_number)
            last_hit_frame = current.frame_number
            last_turn_x = current.x

        previous_direction = direction

    return tuple(hit_frames)


def player_distance_during_frames(
    samples: list[PlayerPositionSample],
    start_frame: int,
    end_frame: int,
) -> tuple[float, float]:
    """Calculate filtered image-plane travel distance for both players."""
    relevant = [
        sample
        for sample in samples
        if start_frame <= sample.frame_number <= end_frame
    ]
    player1_distance = 0.0
    player2_distance = 0.0

    for previous, current in zip(relevant, relevant[1:]):
        frame_gap = current.frame_number - previous.frame_number
        if frame_gap <= 0:
            continue
        maximum_distance = MAX_PLAYER_DISPLACEMENT_PX_PER_FRAME * frame_gap
        player1_step = float(
            np.hypot(
                current.player1[0] - previous.player1[0],
                current.player1[1] - previous.player1[1],
            )
        )
        player2_step = float(
            np.hypot(
                current.player2[0] - previous.player2[0],
                current.player2[1] - previous.player2[1],
            )
        )
        if player1_step <= maximum_distance:
            player1_distance += player1_step
        if player2_step <= maximum_distance:
            player2_distance += player2_step

    return player1_distance, player2_distance


def detect_rallies(
    track_points: list[BallTrackPoint],
    player_samples: list[PlayerPositionSample],
    fps: float,
) -> list[RallyStatistics]:
    """Group nearby ball detections into rallies and calculate their metrics."""
    if not track_points:
        return []

    maximum_gap = max(1, round(fps * RALLY_MAX_BALL_GAP_SECONDS))
    groups: list[list[BallTrackPoint]] = []
    current_group = [track_points[0]]

    for point in track_points[1:]:
        if point.frame_number - current_group[-1].frame_number <= maximum_gap:
            current_group.append(point)
        else:
            groups.append(current_group)
            current_group = [point]
    groups.append(current_group)

    rallies: list[RallyStatistics] = []
    for group in groups:
        start_frame = group[0].frame_number
        end_frame = group[-1].frame_number
        duration = (end_frame - start_frame + 1) / fps
        if len(group) < RALLY_MIN_DETECTIONS or duration < RALLY_MIN_DURATION_SECONDS:
            continue

        hit_frames = estimate_hit_frames(group, fps)
        measured_speeds = [
            float(point.speed) for point in group if point.speed is not None
        ]
        player1_distance, player2_distance = player_distance_during_frames(
            player_samples, start_frame, end_frame
        )
        estimated_hits = len(hit_frames) + 1
        average_speed = float(np.mean(measured_speeds)) if measured_speeds else 0.0
        maximum_speed = max(measured_speeds, default=0.0)
        excitement = (
            duration * 1.5
            + estimated_hits * 1.2
            + maximum_speed / 600.0
            + (player1_distance + player2_distance) / 300.0
        )
        rallies.append(
            RallyStatistics(
                rally_id=len(rallies) + 1,
                start_frame=start_frame,
                end_frame=end_frame,
                duration_seconds=duration,
                ball_detections=len(group),
                estimated_hits=estimated_hits,
                hit_frames=hit_frames,
                average_ball_speed=average_speed,
                maximum_ball_speed=maximum_speed,
                player1_distance=player1_distance,
                player2_distance=player2_distance,
                excitement_score=excitement,
            )
        )

    return rallies


def save_rally_statistics(rallies: list[RallyStatistics]) -> None:
    """Write one analytics row per detected rally."""
    with RALLY_STATISTICS_CSV_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "rally",
                "start_frame",
                "end_frame",
                "duration_seconds",
                "ball_detections",
                "estimated_hits",
                "average_ball_speed_px_s",
                "maximum_ball_speed_px_s",
                "player1_distance_px",
                "player2_distance_px",
                "excitement_score",
            ]
        )
        for rally in rallies:
            writer.writerow(
                [
                    rally.rally_id,
                    rally.start_frame,
                    rally.end_frame,
                    round(rally.duration_seconds, 2),
                    rally.ball_detections,
                    rally.estimated_hits,
                    round(rally.average_ball_speed, 2),
                    round(rally.maximum_ball_speed, 2),
                    round(rally.player1_distance, 2),
                    round(rally.player2_distance, 2),
                    round(rally.excitement_score, 2),
                ]
            )


def create_rally_analysis_chart(rallies: list[RallyStatistics]) -> None:
    """Create a compact four-metric rally comparison dashboard."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    if not rallies:
        for axis in axes.flat:
            axis.axis("off")
        figure.text(
            0.5,
            0.5,
            "No rallies met the minimum duration and detection requirements",
            ha="center",
            va="center",
            fontsize=14,
        )
    else:
        labels = [f"R{rally.rally_id}" for rally in rallies]
        metrics = (
            ([rally.duration_seconds for rally in rallies], "Duration", "seconds"),
            ([rally.estimated_hits for rally in rallies], "Estimated Hits", "hits"),
            (
                [rally.maximum_ball_speed for rally in rallies],
                "Maximum Ball Speed",
                "pixels/second",
            ),
            (
                [rally.excitement_score for rally in rallies],
                "Excitement Score",
                "score",
            ),
        )
        colors = ("tab:blue", "tab:green", "tab:red", "tab:purple")
        for axis, (values, title, ylabel), color in zip(
            axes.flat, metrics, colors
        ):
            axis.bar(labels, values, color=color, alpha=0.85)
            axis.set_title(title)
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.25)

    figure.suptitle("Smart Rally Analysis", fontsize=17)
    figure.tight_layout()
    figure.savefig(RALLY_ANALYSIS_CHART_PATH, dpi=160)
    plt.close(figure)


def apply_rally_overlay(
    video_path: Path,
    valid_source_frames: list[int],
    rallies: list[RallyStatistics],
    fps: float,
    frame_size: tuple[int, int],
) -> None:
    """Add exact rally numbers and cumulative estimated hit counts to the video."""
    if not rallies or not valid_source_frames:
        return

    temporary_path = video_path.with_name(f".{video_path.stem}_rallies.mp4")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not reopen annotated video: {video_path}")
    writer = create_video_writer(temporary_path, fps, frame_size)
    width, _ = frame_size
    rally_index = 0

    try:
        for source_frame in valid_source_frames:
            success, frame = capture.read()
            if not success:
                break

            while (
                rally_index < len(rallies)
                and source_frame > rallies[rally_index].end_frame
            ):
                rally_index += 1

            if rally_index < len(rallies):
                rally = rallies[rally_index]
                if rally.start_frame <= source_frame <= rally.end_frame:
                    hit_count = 1 + bisect_right(rally.hit_frames, source_frame)
                    panel = frame.copy()
                    cv2.rectangle(panel, (width - 520, 18), (width - 18, 112), (0, 0, 0), -1)
                    cv2.addWeighted(panel, 0.62, frame, 0.38, 0, frame)
                    cv2.putText(
                        frame,
                        f"Rally {rally.rally_id}  |  Estimated hits: {hit_count}",
                        (width - 495, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.78,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Excitement: {rally.excitement_score:.1f}",
                        (width - 495, 94),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (0, 220, 255),
                        2,
                        cv2.LINE_AA,
                    )

            writer.write(frame)
    finally:
        capture.release()
        writer.release()

    temporary_path.replace(video_path)


def highlight_title_frame(
    frame_size: tuple[int, int],
    rank: int,
    rally: RallyStatistics,
) -> np.ndarray:
    """Create a title card describing one selected highlight."""
    width, height = frame_size
    frame = np.full((height, width, 3), (18, 22, 31), dtype=np.uint8)
    title = f"Highlight #{rank} - Rally {rally.rally_id}"
    details = (
        f"{rally.duration_seconds:.1f}s  |  ~{rally.estimated_hits} hits  |  "
        f"max {rally.maximum_ball_speed:.0f} px/s"
    )
    for text_value, y, scale, color, thickness in (
        (title, height // 2 - 45, 1.35, (255, 255, 255), 3),
        (details, height // 2 + 25, 0.82, (0, 220, 255), 2),
        (
            f"Excitement score: {rally.excitement_score:.1f}",
            height // 2 + 80,
            0.72,
            (180, 180, 180),
            2,
        ),
    ):
        text_size = cv2.getTextSize(
            text_value, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )[0]
        cv2.putText(
            frame,
            text_value,
            ((width - text_size[0]) // 2, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return frame


def create_highlight_video(
    annotated_video_path: Path,
    valid_source_frames: list[int],
    rallies: list[RallyStatistics],
    fps: float,
    frame_size: tuple[int, int],
) -> int:
    """Create a ranked reel containing the most exciting detected rallies."""
    writer = create_video_writer(HIGHLIGHTS_VIDEO_PATH, fps, frame_size)
    capture = cv2.VideoCapture(str(annotated_video_path))
    if not capture.isOpened():
        writer.release()
        raise OSError(f"Could not open annotated video: {annotated_video_path}")

    selected = sorted(
        rallies, key=lambda rally: rally.excitement_score, reverse=True
    )[:MAX_HIGHLIGHT_RALLIES]
    rank_by_rally = {
        rally.rally_id: rank for rank, rally in enumerate(selected, start=1)
    }
    selected.sort(key=lambda rally: rally.start_frame)
    title_frame_count = max(1, round(fps * HIGHLIGHT_TITLE_SECONDS))
    padding = max(0, round(fps * RALLY_CLIP_PADDING_SECONDS))

    try:
        if not selected:
            empty_rally = RallyStatistics(
                rally_id=0,
                start_frame=0,
                end_frame=0,
                duration_seconds=0.0,
                ball_detections=0,
                estimated_hits=0,
                hit_frames=(),
                average_ball_speed=0.0,
                maximum_ball_speed=0.0,
                player1_distance=0.0,
                player2_distance=0.0,
                excitement_score=0.0,
            )
            title = highlight_title_frame(frame_size, 0, empty_rally)
            cv2.putText(
                title,
                "No rally met the highlight requirements",
                (frame_size[0] // 2 - 390, frame_size[1] // 2 + 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            for _ in range(title_frame_count * 2):
                writer.write(title)
            return 0

        for rally in selected:
            rank = rank_by_rally[rally.rally_id]
            title = highlight_title_frame(frame_size, rank, rally)
            for _ in range(title_frame_count):
                writer.write(title)

            clip_start = rally.start_frame - padding
            clip_end = rally.end_frame + padding
            output_indices = [
                index
                for index, source_frame in enumerate(valid_source_frames)
                if clip_start <= source_frame <= clip_end
            ]
            if not output_indices:
                continue

            capture.set(cv2.CAP_PROP_POS_FRAMES, output_indices[0])
            for _ in range(output_indices[-1] - output_indices[0] + 1):
                success, frame = capture.read()
                if not success:
                    break
                writer.write(frame)
    finally:
        capture.release()
        writer.release()

    return len(selected)


def calculate_match_momentum(
    player_samples: list[PlayerPositionSample],
    track_points: list[BallTrackPoint],
    rallies: list[RallyStatistics],
    score_frame_numbers: list[int],
    player1_scores: list[int],
    player2_scores: list[int],
    fps: float,
) -> list[MomentumSample]:
    """Combine attacking direction, movement, score, and rally intensity."""
    if not player_samples:
        return []

    attack_by_frame: dict[int, float] = {}
    maximum_gap = max(1, round(fps * RALLY_MAX_BALL_GAP_SECONDS))
    for previous, current in zip(track_points, track_points[1:]):
        frame_gap = current.frame_number - previous.frame_number
        if frame_gap <= 0 or frame_gap > maximum_gap:
            continue
        horizontal_direction = 1.0 if current.x > previous.x else -1.0
        speed = current.speed
        if speed is None:
            speed = abs(current.x - previous.x) * fps / frame_gap
        attack_strength = min(speed / 2000.0, 1.0)
        attack_value = horizontal_direction * attack_strength
        for frame_number in range(previous.frame_number + 1, current.frame_number + 1):
            attack_by_frame[frame_number] = attack_value

    score_by_frame = {
        frame_number: (player1_score, player2_score)
        for frame_number, player1_score, player2_score in zip(
            score_frame_numbers, player1_scores, player2_scores
        )
    }
    momentum_samples: list[MomentumSample] = []
    smoothed_player1 = 50.0
    previous_player_sample: PlayerPositionSample | None = None
    last_score = (0, 0)

    for player_sample in player_samples:
        frame_number = player_sample.frame_number
        if frame_number in score_by_frame:
            last_score = score_by_frame[frame_number]
        score_advantage = float(np.clip((last_score[0] - last_score[1]) / 10.0, -1, 1))

        movement_efficiency = 0.0
        if previous_player_sample is not None:
            frame_gap = frame_number - previous_player_sample.frame_number
            if frame_gap > 0:
                player1_step = float(
                    np.hypot(
                        player_sample.player1[0] - previous_player_sample.player1[0],
                        player_sample.player1[1] - previous_player_sample.player1[1],
                    )
                ) / frame_gap
                player2_step = float(
                    np.hypot(
                        player_sample.player2[0] - previous_player_sample.player2[0],
                        player_sample.player2[1] - previous_player_sample.player2[1],
                    )
                ) / frame_gap
                movement_efficiency = float(
                    np.clip((player2_step - player1_step) / 12.0, -1, 1)
                )
        previous_player_sample = player_sample

        rally = next(
            (
                item
                for item in rallies
                if item.start_frame <= frame_number <= item.end_frame
            ),
            None,
        )
        rally_intensity = (
            min(rally.duration_seconds / 6.0, 1.0) if rally is not None else 0.0
        )
        attack_signal = attack_by_frame.get(frame_number, 0.0)
        attack_multiplier = 0.75 + 0.25 * rally_intensity
        raw_player1 = (
            50.0
            + MOMENTUM_ATTACK_WEIGHT * attack_signal * attack_multiplier
            + MOMENTUM_MOVEMENT_WEIGHT * movement_efficiency
            + MOMENTUM_SCORE_WEIGHT * score_advantage
        )
        raw_player1 = float(np.clip(raw_player1, 0.0, 100.0))
        smoothed_player1 = (
            MOMENTUM_SMOOTHING_ALPHA * raw_player1
            + (1.0 - MOMENTUM_SMOOTHING_ALPHA) * smoothed_player1
        )
        momentum_samples.append(
            MomentumSample(
                frame_number=frame_number,
                player1_momentum=smoothed_player1,
                player2_momentum=100.0 - smoothed_player1,
                score_advantage=score_advantage,
                attack_signal=attack_signal,
                movement_efficiency=movement_efficiency,
                rally_intensity=rally_intensity,
                rally_id=None if rally is None else rally.rally_id,
            )
        )

    return momentum_samples


def save_match_momentum(momentum_samples: list[MomentumSample]) -> None:
    """Write the momentum timeline and all explainable input components."""
    with MOMENTUM_CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "frame",
                "player1_momentum",
                "player2_momentum",
                "score_advantage",
                "attack_signal",
                "movement_efficiency",
                "rally_intensity",
                "rally",
            ]
        )
        for sample in momentum_samples:
            writer.writerow(
                [
                    sample.frame_number,
                    round(sample.player1_momentum, 2),
                    round(sample.player2_momentum, 2),
                    round(sample.score_advantage, 4),
                    round(sample.attack_signal, 4),
                    round(sample.movement_efficiency, 4),
                    round(sample.rally_intensity, 4),
                    "" if sample.rally_id is None else sample.rally_id,
                ]
            )


def create_momentum_chart(
    momentum_samples: list[MomentumSample], rallies: list[RallyStatistics]
) -> None:
    """Plot both players' explainable 0-100 momentum indices over time."""
    figure, axis = plt.subplots(figsize=(14, 7))
    if momentum_samples:
        frames = [sample.frame_number for sample in momentum_samples]
        player1 = [sample.player1_momentum for sample in momentum_samples]
        player2 = [sample.player2_momentum for sample in momentum_samples]
        axis.plot(frames, player1, label="Player 1 (left)", color="tab:blue", linewidth=2)
        axis.plot(frames, player2, label="Player 2 (right)", color="tab:orange", linewidth=2)
        axis.fill_between(
            frames,
            player1,
            player2,
            where=np.asarray(player1) >= np.asarray(player2),
            color="tab:blue",
            alpha=0.10,
            interpolate=True,
        )
        axis.fill_between(
            frames,
            player1,
            player2,
            where=np.asarray(player2) > np.asarray(player1),
            color="tab:orange",
            alpha=0.10,
            interpolate=True,
        )
        for rally in rallies:
            axis.axvspan(
                rally.start_frame,
                rally.end_frame,
                color="gray",
                alpha=0.08,
            )
            axis.text(
                (rally.start_frame + rally.end_frame) / 2,
                97,
                f"R{rally.rally_id}",
                ha="center",
                va="top",
                fontsize=9,
            )
        axis.legend(loc="lower right")
    else:
        axis.text(
            0.5,
            0.5,
            "No valid player-position samples were available",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    axis.axhline(50, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axis.set_ylim(0, 100)
    axis.set_title("Match Momentum Timeline (Explainable Heuristic)")
    axis.set_xlabel("Source Frame")
    axis.set_ylabel("Momentum Index")
    axis.grid(alpha=0.25)
    figure.text(
        0.5,
        0.015,
        "Combines ball direction/speed, rally duration, movement efficiency, and score; not win probability.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(MOMENTUM_CHART_PATH, dpi=160)
    plt.close(figure)


def create_html_dashboard(
    inspected_frames: int,
    written_frames: int,
    invalid_frames: int,
    ball_detection_frames: int,
    fastest_ball_speed: float,
    rallies: list[RallyStatistics],
    highlight_count: int,
    momentum_samples: list[MomentumSample],
    ball_track_points: list[BallTrackPoint],
    player_position_samples: list[PlayerPositionSample],
    valid_source_frames: list[int],
    score_frame_numbers: list[int],
    player1_scores: list[int],
    player2_scores: list[int],
    fps: float,
) -> None:
    """Generate a responsive, self-contained UI around all analysis artifacts."""
    total_estimated_hits = sum(rally.estimated_hits for rally in rallies)
    detection_rate = (
        ball_detection_frames / written_frames * 100 if written_frames else 0.0
    )
    best_rally = max(rallies, key=lambda rally: rally.excitement_score, default=None)
    best_rally_text = (
        f"Rally {best_rally.rally_id} · {best_rally.excitement_score:.1f} score"
        if best_rally is not None
        else "No qualifying rally"
    )
    momentum_swing = max(
        (
            abs(sample.player1_momentum - sample.player2_momentum)
            for sample in momentum_samples
        ),
        default=0.0,
    )
    ball_by_frame = {point.frame_number: point for point in ball_track_points}
    players_by_frame = {
        sample.frame_number: sample for sample in player_position_samples
    }
    momentum_by_frame = {
        sample.frame_number: sample for sample in momentum_samples
    }
    scores_by_frame = {
        frame_number: (player1_score, player2_score)
        for frame_number, player1_score, player2_score in zip(
            score_frame_numbers, player1_scores, player2_scores
        )
    }
    timeline_data: list[dict[str, Any]] = []
    last_score_values: tuple[int, int] | None = None
    rally_index = 0
    for source_frame in valid_source_frames:
        if source_frame in scores_by_frame:
            last_score_values = scores_by_frame[source_frame]
        while (
            rally_index < len(rallies)
            and source_frame > rallies[rally_index].end_frame
        ):
            rally_index += 1
        rally = None
        if rally_index < len(rallies):
            candidate_rally = rallies[rally_index]
            if candidate_rally.start_frame <= source_frame <= candidate_rally.end_frame:
                rally = candidate_rally

        ball = ball_by_frame.get(source_frame)
        players = players_by_frame.get(source_frame)
        momentum = momentum_by_frame.get(source_frame)
        timeline_data.append(
            {
                "f": source_frame,
                "bx": None if ball is None else round(ball.x, 2),
                "by": None if ball is None else round(ball.y, 2),
                "bs": (
                    None
                    if ball is None or ball.speed is None
                    else round(ball.speed, 2)
                ),
                "p1": (
                    None
                    if players is None
                    else [round(players.player1[0], 2), round(players.player1[1], 2)]
                ),
                "p2": (
                    None
                    if players is None
                    else [round(players.player2[0], 2), round(players.player2[1], 2)]
                ),
                "m1": (
                    None if momentum is None else round(momentum.player1_momentum, 2)
                ),
                "m2": (
                    None if momentum is None else round(momentum.player2_momentum, 2)
                ),
                "r": 0 if rally is None else rally.rally_id,
                "h": (
                    0
                    if rally is None
                    else 1 + bisect_right(rally.hit_frames, source_frame)
                ),
                "s1": None if last_score_values is None else last_score_values[0],
                "s2": None if last_score_values is None else last_score_values[1],
            }
        )
    timeline_json = json.dumps(timeline_data, separators=(",", ":"))

    if rallies:
        rally_rows = "\n".join(
            f"""
              <tr>
                <td><span class="rally-pill">R{rally.rally_id}</span></td>
                <td>{rally.start_frame}–{rally.end_frame}</td>
                <td>{rally.duration_seconds:.2f}s</td>
                <td>~{rally.estimated_hits}</td>
                <td>{rally.average_ball_speed:.0f} px/s</td>
                <td>{rally.maximum_ball_speed:.0f} px/s</td>
                <td>{rally.excitement_score:.1f}</td>
              </tr>"""
            for rally in rallies
        )
    else:
        rally_rows = """
              <tr>
                <td colspan="7" class="empty-state">No rally met the minimum analysis requirements.</td>
              </tr>"""

    dashboard = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <meta name="description" content="Interactive table-tennis match intelligence dashboard with pose, ball, rally, score, and momentum analysis.">
  <title>RallyIQ · Table Tennis Match Intelligence</title>
  <style>
    :root {
      --bg: #070b12;
      --bg-soft: #0a101a;
      --panel: rgba(15, 23, 36, 0.92);
      --panel-strong: #111b2a;
      --line: rgba(163, 186, 215, 0.16);
      --text: #f3f7fb;
      --muted: #91a2b7;
      --cyan: #4ce6c3;
      --cyan-soft: rgba(76, 230, 195, 0.14);
      --amber: #ffcc66;
      --blue: #5d8cff;
      --danger: #ff6b78;
      --radius: 22px;
      --shadow: 0 24px 70px rgba(0, 0, 0, 0.34);
      --content: min(1440px, calc(100% - 40px));
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-width: 320px;
      background:
        radial-gradient(circle at 8% 8%, rgba(76, 230, 195, 0.10), transparent 24rem),
        radial-gradient(circle at 92% 18%, rgba(93, 140, 255, 0.12), transparent 28rem),
        var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }

    a { color: inherit; text-decoration: none; }
    button, a { -webkit-tap-highlight-color: transparent; }
    button { font: inherit; }
    :focus-visible { outline: 3px solid var(--cyan); outline-offset: 3px; }

    .site-header {
      position: sticky;
      top: 0;
      z-index: 20;
      border-bottom: 1px solid var(--line);
      background: rgba(7, 11, 18, 0.82);
      backdrop-filter: blur(18px);
    }

    .nav {
      width: var(--content);
      min-height: 68px;
      margin: auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
    }

    .brand {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      font-size: 1.08rem;
      font-weight: 850;
      letter-spacing: 0.04em;
    }

    .brand-mark {
      width: 34px;
      height: 34px;
      display: grid;
      place-items: center;
      border: 1px solid rgba(76, 230, 195, 0.5);
      border-radius: 10px;
      background: var(--cyan-soft);
      color: var(--cyan);
      font-size: 0.74rem;
    }

    .brand span { color: var(--cyan); }
    .nav-links { display: flex; align-items: center; gap: 28px; }
    .nav-links a { color: var(--muted); font-size: 0.9rem; font-weight: 650; }
    .nav-links a:hover { color: var(--text); }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      font-size: 0.8rem;
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--cyan);
      box-shadow: 0 0 16px var(--cyan);
    }

    main { width: var(--content); margin: auto; }
    .hero {
      position: relative;
      overflow: hidden;
      padding: 82px 0 54px;
      border-bottom: 1px solid var(--line);
    }

    .hero::after {
      content: "";
      position: absolute;
      width: min(580px, 56vw);
      height: 260px;
      right: -110px;
      top: 42px;
      border: 1px solid rgba(76, 230, 195, 0.14);
      border-radius: 18px;
      transform: perspective(500px) rotateX(58deg) rotateZ(-8deg);
      box-shadow: inset 0 0 80px rgba(76, 230, 195, 0.04);
      pointer-events: none;
    }

    .eyebrow {
      margin: 0 0 16px;
      color: var(--cyan);
      font-size: 0.76rem;
      font-weight: 800;
      letter-spacing: 0.18em;
      text-transform: uppercase;
    }

    h1 {
      max-width: 830px;
      margin: 0;
      font-size: clamp(2.7rem, 7vw, 6.4rem);
      line-height: 0.96;
      letter-spacing: -0.065em;
    }

    h1 em { color: var(--cyan); font-style: normal; }
    .hero-copy {
      max-width: 690px;
      margin: 24px 0 0;
      color: var(--muted);
      font-size: clamp(1rem, 1.7vw, 1.18rem);
    }

    .hero-actions { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 30px; }
    .button {
      min-height: 46px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 0 18px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.03);
      color: var(--text);
      font-size: 0.9rem;
      font-weight: 750;
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
    }

    .button:hover { transform: translateY(-2px); border-color: rgba(76, 230, 195, 0.5); }
    .button.primary { background: var(--cyan); color: #04110e; border-color: var(--cyan); }
    .hero-meta { display: flex; flex-wrap: wrap; gap: 24px; margin-top: 34px; color: var(--muted); font-size: 0.82rem; }
    .hero-meta strong { color: var(--text); }

    section { padding: 64px 0; }
    .section-heading { display: flex; align-items: end; justify-content: space-between; gap: 24px; margin-bottom: 24px; }
    .section-heading h2 { margin: 0; font-size: clamp(1.7rem, 3vw, 2.5rem); letter-spacing: -0.035em; }
    .section-heading p { max-width: 620px; margin: 8px 0 0; color: var(--muted); }

    .metrics-grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 14px; }
    .metric {
      min-height: 154px;
      padding: 21px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: linear-gradient(145deg, rgba(17, 27, 42, 0.95), rgba(10, 16, 26, 0.9));
      box-shadow: 0 16px 34px rgba(0, 0, 0, 0.16);
    }

    .metric-label { color: var(--muted); font-size: 0.76rem; font-weight: 750; letter-spacing: 0.08em; text-transform: uppercase; }
    .metric-value { margin-top: 16px; font-size: clamp(1.65rem, 2.8vw, 2.35rem); font-weight: 850; letter-spacing: -0.045em; }
    .metric-note { margin-top: 7px; color: var(--muted); font-size: 0.78rem; }
    .metric.accent { border-color: rgba(76, 230, 195, 0.34); background: linear-gradient(145deg, rgba(21, 60, 58, 0.65), rgba(10, 20, 28, 0.92)); }
    .metric.accent .metric-value { color: var(--cyan); }

    .media-grid { display: grid; grid-template-columns: 1fr; gap: 18px; }
    .card {
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel);
      box-shadow: var(--shadow);
    }

    .card-header { display: flex; align-items: center; justify-content: space-between; gap: 14px; padding: 18px 20px; border-bottom: 1px solid var(--line); }
    .card-kicker { color: var(--cyan); font-size: 0.7rem; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase; }
    .card-title { margin: 4px 0 0; font-size: 1rem; }
    .card-badge { padding: 6px 9px; border-radius: 999px; background: var(--cyan-soft); color: var(--cyan); font-size: 0.72rem; font-weight: 750; white-space: nowrap; }
    .video-wrap { aspect-ratio: 16 / 9; background: #000; }
    video { display: block; width: 100%; height: 100%; object-fit: contain; }
    .video-note { padding: 14px 18px 18px; color: var(--muted); font-size: 0.82rem; }

    .timeline-shell { padding: 18px; border-top: 1px solid var(--line); background: rgba(5, 10, 17, 0.72); }
    .timeline-heading { display: flex; align-items: end; justify-content: space-between; gap: 18px; margin-bottom: 14px; }
    .timeline-heading h4 { margin: 0; font-size: 1rem; }
    .timeline-heading p { max-width: 560px; margin: 0; color: var(--muted); font-size: 0.78rem; }
    .timeline-live-grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 8px; margin-bottom: 12px; }
    .live-stat { min-width: 0; padding: 11px 12px; border: 1px solid var(--line); border-radius: 11px; background: rgba(255, 255, 255, 0.025); }
    .live-label { display: block; color: var(--muted); font-size: 0.62rem; font-weight: 800; letter-spacing: 0.09em; text-transform: uppercase; }
    .live-value { display: block; overflow: hidden; margin-top: 5px; color: var(--text); font-size: 0.8rem; font-weight: 750; text-overflow: ellipsis; white-space: nowrap; }
    .live-stat.primary .live-value { color: var(--cyan); }
    .timeline-canvas-wrap { position: relative; overflow: hidden; border: 1px solid var(--line); border-radius: 12px; background: #090f18; cursor: crosshair; touch-action: none; }
    #match-timeline { display: block; width: 100%; height: 168px; }
    .timeline-scrubber { width: 100%; height: 26px; margin: 8px 0 0; accent-color: var(--cyan); cursor: pointer; }
    .timeline-legend { display: flex; flex-wrap: wrap; gap: 10px 18px; color: var(--muted); font-size: 0.72rem; }
    .legend-item { display: inline-flex; align-items: center; gap: 7px; }
    .legend-swatch { width: 18px; height: 3px; border-radius: 999px; background: var(--cyan); }
    .legend-swatch.player-two { background: var(--blue); }
    .legend-swatch.speed { background: var(--amber); }
    .legend-swatch.rally { height: 10px; background: rgba(76, 230, 195, 0.2); border: 1px solid rgba(76, 230, 195, 0.42); }

    .chart-layout { display: grid; grid-template-columns: 220px minmax(0, 1fr); gap: 18px; }
    .chart-tabs { display: flex; flex-direction: column; gap: 8px; }
    .chart-tab {
      width: 100%;
      padding: 14px 15px;
      border: 1px solid transparent;
      border-radius: 12px;
      background: transparent;
      color: var(--muted);
      text-align: left;
      cursor: pointer;
    }

    .chart-tab:hover { color: var(--text); background: rgba(255,255,255,0.035); }
    .chart-tab[aria-pressed="true"] { color: var(--text); border-color: rgba(76, 230, 195, 0.32); background: var(--cyan-soft); }
    .chart-panel { min-width: 0; }
    .chart-frame { min-height: 420px; display: grid; place-items: center; overflow: hidden; border: 1px solid var(--line); border-radius: var(--radius); background: #f7f8fa; }
    .chart-frame img { display: block; width: 100%; max-height: 720px; object-fit: contain; transition: opacity 140ms ease, transform 140ms ease; }
    .chart-frame img.is-switching { opacity: 0; transform: scale(0.99); }
    .chart-meta { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-top: 12px; color: var(--muted); font-size: 0.82rem; }

    .data-grid { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(300px, 0.65fr); gap: 18px; }
    .table-card { overflow: hidden; }
    .table-scroll { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; min-width: 780px; }
    th, td { padding: 15px 17px; border-bottom: 1px solid var(--line); text-align: left; font-size: 0.84rem; white-space: nowrap; }
    th { color: var(--muted); font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase; }
    tbody tr:hover { background: rgba(255,255,255,0.025); }
    .rally-pill { display: inline-flex; padding: 5px 8px; border-radius: 8px; background: var(--cyan-soft); color: var(--cyan); font-weight: 800; }
    .empty-state { color: var(--muted); text-align: center; padding: 32px; }

    .download-list { display: grid; gap: 8px; padding: 16px; }
    .download-item { display: flex; justify-content: space-between; gap: 16px; padding: 14px; border: 1px solid var(--line); border-radius: 12px; background: rgba(255,255,255,0.02); }
    .download-item:hover { border-color: rgba(76, 230, 195, 0.42); }
    .download-name { font-size: 0.86rem; font-weight: 700; }
    .download-type { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; }

    .methodology { margin-top: 18px; border: 1px solid var(--line); border-radius: 16px; background: rgba(255,255,255,0.02); }
    .methodology summary { padding: 17px 19px; cursor: pointer; font-weight: 750; }
    .methodology-content { padding: 0 19px 18px; color: var(--muted); font-size: 0.88rem; }
    .methodology-content strong { color: var(--text); }

    footer { width: var(--content); margin: 0 auto; padding: 30px 0 42px; border-top: 1px solid var(--line); color: var(--muted); font-size: 0.8rem; display: flex; justify-content: space-between; gap: 20px; }

    @media (max-width: 1160px) {
      .metrics-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .media-grid, .data-grid { grid-template-columns: 1fr; }
      .timeline-live-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }

    @media (max-width: 780px) {
      :root { --content: min(100% - 24px, 1440px); --radius: 17px; }
      .nav { min-height: 60px; }
      .nav-links { display: none; }
      .status { padding: 7px 9px; }
      .hero { padding: 56px 0 40px; }
      .hero::after { opacity: 0.45; }
      .metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .metric { min-height: 132px; padding: 17px; }
      section { padding: 44px 0; }
      .section-heading { align-items: start; flex-direction: column; }
      .media-grid { grid-template-columns: 1fr; }
      .chart-layout { grid-template-columns: 1fr; }
      .chart-tabs { flex-direction: row; overflow-x: auto; padding-bottom: 5px; }
      .chart-tab { width: auto; flex: 0 0 auto; white-space: nowrap; }
      .chart-frame { min-height: 260px; }
      .chart-meta { align-items: start; flex-direction: column; }
      .timeline-heading { align-items: start; flex-direction: column; }
      #match-timeline { height: 145px; }
      footer { flex-direction: column; }
    }

    @media (max-width: 460px) {
      .brand-word { display: none; }
      .status-label { display: none; }
      .metrics-grid { grid-template-columns: 1fr 1fr; gap: 10px; }
      .metric-value { font-size: 1.55rem; }
      .hero-actions .button { width: 100%; }
      .card-header { align-items: flex-start; }
      .timeline-shell { padding: 13px; }
      .timeline-live-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      #match-timeline { height: 125px; }
    }

    @media (prefers-reduced-motion: reduce) {
      html { scroll-behavior: auto; }
      *, *::before, *::after { transition-duration: 0.01ms !important; }
    }
  </style>
</head>
<body>
  <header class="site-header">
    <nav class="nav" aria-label="Dashboard navigation">
      <a class="brand" href="#top" aria-label="RallyIQ dashboard home">
        <span class="brand-mark">RIQ</span>
        <span class="brand-word">RALLY<span>IQ</span></span>
      </a>
      <div class="nav-links">
        <a href="#overview">Overview</a>
        <a href="#playback">Playback</a>
        <a href="#insights">Insights</a>
        <a href="#data">Data</a>
      </div>
      <div class="status"><span class="status-dot"></span><span class="status-label">Analysis ready</span></div>
    </nav>
  </header>

  <main id="top">
    <section class="hero" aria-labelledby="dashboard-title">
      <p class="eyebrow">Computer vision · Match intelligence</p>
      <h1 id="dashboard-title">See the match.<br><em>Read the momentum.</em></h1>
      <p class="hero-copy">One view for pose tracking, player movement, ball trajectory, rally intensity, highlights, score state, and explainable momentum.</p>
      <div class="hero-actions">
        <a class="button primary" href="#playback">Watch analysis</a>
        <a class="button" href="match_highlights.mp4" download>Download highlights</a>
      </div>
      <div class="hero-meta">
        <span><strong>Frame range</strong> %%FRAME_RANGE%%</span>
        <span><strong>Models</strong> YOLO Pose + RF-DETR</span>
        <span><strong>Best sequence</strong> %%BEST_RALLY%%</span>
      </div>
    </section>

    <section id="overview" aria-labelledby="overview-title">
      <div class="section-heading">
        <div><p class="eyebrow">Match snapshot</p><h2 id="overview-title">The signal at a glance</h2></div>
        <p>Metrics are calculated from valid broadcast frames only. Speeds and travel distances use image-plane pixels.</p>
      </div>
      <div class="metrics-grid">
        <article class="metric"><div class="metric-label">Valid frames</div><div class="metric-value">%%VALID_FRAMES%%</div><div class="metric-note">%%INVALID_FRAMES%% camera cut skipped</div></article>
        <article class="metric accent"><div class="metric-label">Ball detections</div><div class="metric-value">%%BALL_DETECTIONS%%</div><div class="metric-note">%%DETECTION_RATE%%% of valid frames</div></article>
        <article class="metric"><div class="metric-label">Rallies</div><div class="metric-value">%%RALLY_COUNT%%</div><div class="metric-note">gap-stabilized sequences</div></article>
        <article class="metric"><div class="metric-label">Estimated hits</div><div class="metric-value">~%%TOTAL_HITS%%</div><div class="metric-note">direction-reversal estimate</div></article>
        <article class="metric"><div class="metric-label">Fastest ball</div><div class="metric-value">%%FASTEST_SPEED%%</div><div class="metric-note">pixels per second</div></article>
        <article class="metric"><div class="metric-label">Momentum swing</div><div class="metric-value">%%MOMENTUM_SWING%%</div><div class="metric-note">point spread · heuristic</div></article>
      </div>
    </section>

    <section id="playback" aria-labelledby="playback-title">
      <div class="section-heading">
        <div><p class="eyebrow">Playback</p><h2 id="playback-title">Analysis in motion</h2></div>
        <p>Follow player poses, the speed-colored ball trail, live rally identification, and estimated contact count.</p>
      </div>
      <div class="media-grid">
        <article class="card">
          <div class="card-header"><div><div class="card-kicker">Full analysis</div><h3 class="card-title">Annotated match segment</h3></div><span class="card-badge">%%VALID_FRAMES%% frames</span></div>
          <div class="video-wrap"><video id="analysis-video" controls playsinline preload="metadata"><source src="output_with_detections.mp4" type="video/mp4">Your browser does not support MP4 playback.</video></div>
          <div class="timeline-shell">
            <div class="timeline-heading">
              <h4>Interactive match timeline</h4>
              <p>Play, scrub, or click the graph. The readouts stay synchronized with the processed-video frame.</p>
            </div>
            <div class="timeline-live-grid" aria-live="polite">
              <div class="live-stat primary"><span class="live-label">Source frame</span><span class="live-value" id="timeline-frame">—</span></div>
              <div class="live-stat"><span class="live-label">Rally / hits</span><span class="live-value" id="timeline-rally">—</span></div>
              <div class="live-stat"><span class="live-label">Ball speed</span><span class="live-value" id="timeline-speed">—</span></div>
              <div class="live-stat"><span class="live-label">Momentum</span><span class="live-value" id="timeline-momentum">—</span></div>
              <div class="live-stat"><span class="live-label">Score</span><span class="live-value" id="timeline-score">—</span></div>
              <div class="live-stat"><span class="live-label">Tracked positions</span><span class="live-value" id="timeline-positions">—</span></div>
            </div>
            <div class="timeline-canvas-wrap" id="timeline-canvas-wrap" title="Click to seek">
              <canvas id="match-timeline" aria-label="Synchronized chart showing player momentum, ball speed, rally spans, and the current video position"></canvas>
            </div>
            <input class="timeline-scrubber" id="timeline-scrubber" type="range" min="0" value="0" step="1" aria-label="Seek through processed video frames">
            <div class="timeline-legend" aria-label="Timeline legend">
              <span class="legend-item"><span class="legend-swatch"></span>Player 1 momentum</span>
              <span class="legend-item"><span class="legend-swatch player-two"></span>Player 2 momentum</span>
              <span class="legend-item"><span class="legend-swatch speed"></span>Ball speed</span>
              <span class="legend-item"><span class="legend-swatch rally"></span>Detected rally</span>
            </div>
          </div>
          <div class="video-note">Pose skeletons, player boxes, ball confidence, speed trail, and rally intelligence are rendered directly into the video.</div>
        </article>
        <article class="card">
          <div class="card-header"><div><div class="card-kicker">Auto-curated</div><h3 class="card-title">Top rally highlights</h3></div><span class="card-badge">%%HIGHLIGHT_COUNT%% selected</span></div>
          <div class="video-wrap"><video controls playsinline preload="metadata"><source src="match_highlights.mp4" type="video/mp4">Your browser does not support MP4 playback.</video></div>
          <div class="video-note">Rallies are ranked by duration, estimated hits, peak speed, and combined player movement.</div>
        </article>
      </div>
    </section>

    <section id="insights" aria-labelledby="insights-title">
      <div class="section-heading">
        <div><p class="eyebrow">Visual intelligence</p><h2 id="insights-title">Explore every analytical layer</h2></div>
        <p>Switch between charts without leaving the dashboard. Each image can also be downloaded at full resolution.</p>
      </div>
      <div class="chart-layout">
        <div class="chart-tabs" role="group" aria-label="Select analysis chart">
          <button class="chart-tab" type="button" aria-pressed="true" data-src="match_momentum.png" data-title="Match momentum" data-alt="Two-line explainable momentum timeline for both players">Momentum</button>
          <button class="chart-tab" type="button" aria-pressed="false" data-src="player_position_heatmap.png" data-title="Player position density" data-alt="Court image with combined player position density heatmap">Player heatmap</button>
          <button class="chart-tab" type="button" aria-pressed="false" data-src="ball_speed_over_time.png" data-title="Ball speed over time" data-alt="Line chart of smoothed table-tennis ball speed over source frames">Ball speed</button>
          <button class="chart-tab" type="button" aria-pressed="false" data-src="rally_analysis.png" data-title="Smart rally analysis" data-alt="Rally comparison dashboard for duration, hits, speed, and excitement">Rally analysis</button>
          <button class="chart-tab" type="button" aria-pressed="false" data-src="score_over_time.png" data-title="Score progression" data-alt="Player calculated score progression over source frames">Score</button>
        </div>
        <div class="chart-panel">
          <div class="chart-frame"><img id="active-chart" src="match_momentum.png" alt="Two-line explainable momentum timeline for both players"></div>
          <div class="chart-meta"><span id="chart-caption" aria-live="polite">Match momentum</span><a id="chart-download" class="button" href="match_momentum.png" download>Download chart</a></div>
        </div>
      </div>
    </section>

    <section id="data" aria-labelledby="data-title">
      <div class="section-heading">
        <div><p class="eyebrow">Structured data</p><h2 id="data-title">Rally table & exports</h2></div>
        <p>Use the CSV exports for your report, additional charts, or future model experiments.</p>
      </div>
      <div class="data-grid">
        <article class="card table-card">
          <div class="card-header"><div><div class="card-kicker">Rally ranking</div><h3 class="card-title">Detected sequences</h3></div><span class="card-badge">%%RALLY_COUNT%% total</span></div>
          <div class="table-scroll">
            <table>
              <thead><tr><th>Rally</th><th>Frames</th><th>Duration</th><th>Hits</th><th>Average speed</th><th>Peak speed</th><th>Excitement</th></tr></thead>
              <tbody>%%RALLY_ROWS%%</tbody>
            </table>
          </div>
        </article>
        <aside class="card">
          <div class="card-header"><div><div class="card-kicker">Exports</div><h3 class="card-title">Download datasets</h3></div></div>
          <div class="download-list">
            <a class="download-item" href="player_positions.csv" download><span class="download-name">Player positions</span><span class="download-type">CSV</span></a>
            <a class="download-item" href="ball_positions.csv" download><span class="download-name">Ball tracking</span><span class="download-type">CSV</span></a>
            <a class="download-item" href="rally_statistics.csv" download><span class="download-name">Rally statistics</span><span class="download-type">CSV</span></a>
            <a class="download-item" href="match_momentum.csv" download><span class="download-name">Momentum components</span><span class="download-type">CSV</span></a>
          </div>
        </aside>
      </div>

      <details class="methodology">
        <summary>How should these metrics be interpreted?</summary>
        <div class="methodology-content">
          <p><strong>Ball speed</strong> is measured in pixels per second, not km/h. Real-world speed requires camera calibration and court-plane perspective correction.</p>
          <p><strong>Estimated hits</strong> use stable horizontal ball-direction reversals and should be treated as an approximation when detections are missed.</p>
          <p><strong>Momentum</strong> combines ball direction and speed, rally duration, movement efficiency, and available score state. It is an explainable analytical index—not win probability.</p>
        </div>
      </details>
    </section>
  </main>

  <footer><span>RallyIQ · Table Tennis Analyzer</span><span>Generated from %%INSPECTED_FRAMES%% source frames · local analysis dashboard</span></footer>

  <script>
    const timelineData = %%TIMELINE_DATA%%;
    const analysisFps = %%FPS%%;
    const analysisVideo = document.getElementById('analysis-video');
    const timelineCanvas = document.getElementById('match-timeline');
    const timelineCanvasWrap = document.getElementById('timeline-canvas-wrap');
    const timelineScrubber = document.getElementById('timeline-scrubber');
    const timelineFields = {
      frame: document.getElementById('timeline-frame'),
      rally: document.getElementById('timeline-rally'),
      speed: document.getElementById('timeline-speed'),
      momentum: document.getElementById('timeline-momentum'),
      score: document.getElementById('timeline-score'),
      positions: document.getElementById('timeline-positions'),
    };
    let activeTimelineIndex = 0;
    let animationFrameId = null;

    timelineScrubber.max = String(Math.max(0, timelineData.length - 1));

    function clampTimelineIndex(index) {
      return Math.max(0, Math.min(timelineData.length - 1, Number(index) || 0));
    }

    function formatPoint(point) {
      return point ? `${Math.round(point[0])},${Math.round(point[1])}` : '—';
    }

    function drawTimeline() {
      if (!timelineData.length) return;
      const rect = timelineCanvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(1, rect.width);
      const height = Math.max(1, rect.height);
      const pixelWidth = Math.round(width * dpr);
      const pixelHeight = Math.round(height * dpr);
      if (timelineCanvas.width !== pixelWidth || timelineCanvas.height !== pixelHeight) {
        timelineCanvas.width = pixelWidth;
        timelineCanvas.height = pixelHeight;
      }
      const context = timelineCanvas.getContext('2d');
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, width, height);

      const inset = 10;
      const plotWidth = Math.max(1, width - inset * 2);
      const plotHeight = Math.max(1, height - inset * 2);
      const xAt = (index) => inset + (index / Math.max(1, timelineData.length - 1)) * plotWidth;
      const yAt = (value) => inset + (1 - Math.max(0, Math.min(100, value)) / 100) * plotHeight;

      context.strokeStyle = 'rgba(163,186,215,0.12)';
      context.lineWidth = 1;
      [0.25, 0.5, 0.75].forEach((fraction) => {
        const y = inset + plotHeight * fraction;
        context.beginPath();
        context.moveTo(inset, y);
        context.lineTo(width - inset, y);
        context.stroke();
      });

      let bandStart = null;
      let bandRally = 0;
      timelineData.forEach((sample, index) => {
        if (sample.r !== bandRally) {
          if (bandRally && bandStart !== null) {
            context.fillStyle = 'rgba(76,230,195,0.10)';
            context.fillRect(xAt(bandStart), inset, Math.max(2, xAt(index) - xAt(bandStart)), plotHeight);
          }
          bandRally = sample.r;
          bandStart = sample.r ? index : null;
        }
      });
      if (bandRally && bandStart !== null) {
        context.fillStyle = 'rgba(76,230,195,0.10)';
        context.fillRect(xAt(bandStart), inset, width - inset - xAt(bandStart), plotHeight);
      }

      function drawSeries(valueFor, color, lineWidth = 2) {
        context.strokeStyle = color;
        context.lineWidth = lineWidth;
        context.lineJoin = 'round';
        context.lineCap = 'round';
        context.beginPath();
        let drawing = false;
        timelineData.forEach((sample, index) => {
          const value = valueFor(sample);
          if (value === null || value === undefined) {
            drawing = false;
            return;
          }
          const x = xAt(index);
          const y = yAt(value);
          if (drawing) context.lineTo(x, y);
          else context.moveTo(x, y);
          drawing = true;
        });
        context.stroke();
      }

      const maxSpeed = Math.max(1, ...timelineData.map((sample) => sample.bs || 0));
      drawSeries((sample) => sample.m1, '#4ce6c3', 2.2);
      drawSeries((sample) => sample.m2, '#5d8cff', 2.2);
      drawSeries((sample) => sample.bs === null ? null : sample.bs / maxSpeed * 100, '#ffcc66', 1.5);

      const playheadX = xAt(activeTimelineIndex);
      context.strokeStyle = 'rgba(255,255,255,0.95)';
      context.lineWidth = 1.5;
      context.beginPath();
      context.moveTo(playheadX, 0);
      context.lineTo(playheadX, height);
      context.stroke();
      context.fillStyle = '#ffffff';
      context.beginPath();
      context.arc(playheadX, 8, 3.5, 0, Math.PI * 2);
      context.fill();
    }

    function updateTimeline(index) {
      if (!timelineData.length) return;
      activeTimelineIndex = Math.round(clampTimelineIndex(index));
      const sample = timelineData[activeTimelineIndex];
      timelineScrubber.value = String(activeTimelineIndex);
      timelineFields.frame.textContent = String(sample.f);
      timelineFields.rally.textContent = sample.r ? `R${sample.r} · ~${sample.h} hits` : 'Between rallies';
      timelineFields.speed.textContent = sample.bs === null ? 'Not detected' : `${sample.bs.toFixed(0)} px/s`;
      timelineFields.momentum.textContent = sample.m1 === null ? 'Not available' : `P1 ${sample.m1.toFixed(1)} · P2 ${sample.m2.toFixed(1)}`;
      timelineFields.score.textContent = sample.s1 === null ? 'Not read' : `${sample.s1}–${sample.s2}`;
      const ball = sample.bx === null ? 'ball —' : `ball ${Math.round(sample.bx)},${Math.round(sample.by)}`;
      timelineFields.positions.textContent = `L ${formatPoint(sample.p1)} · R ${formatPoint(sample.p2)} · ${ball}`;
      drawTimeline();
    }

    function syncTimelineToVideo() {
      updateTimeline(Math.floor(analysisVideo.currentTime * analysisFps));
    }

    function followPlayback() {
      syncTimelineToVideo();
      if (!analysisVideo.paused && !analysisVideo.ended) {
        animationFrameId = window.requestAnimationFrame(followPlayback);
      }
    }

    analysisVideo.addEventListener('play', () => {
      if (animationFrameId !== null) window.cancelAnimationFrame(animationFrameId);
      followPlayback();
    });
    ['pause', 'seeked', 'timeupdate', 'loadedmetadata'].forEach((eventName) => {
      analysisVideo.addEventListener(eventName, syncTimelineToVideo);
    });
    timelineScrubber.addEventListener('input', () => {
      const index = clampTimelineIndex(timelineScrubber.value);
      analysisVideo.currentTime = index / analysisFps;
      updateTimeline(index);
    });
    timelineCanvasWrap.addEventListener('pointerdown', (event) => {
      const rect = timelineCanvasWrap.getBoundingClientRect();
      const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
      const index = Math.round(fraction * Math.max(0, timelineData.length - 1));
      analysisVideo.currentTime = index / analysisFps;
      updateTimeline(index);
    });
    new ResizeObserver(drawTimeline).observe(timelineCanvasWrap);
    updateTimeline(0);

    const chartButtons = [...document.querySelectorAll('.chart-tab')];
    const chartImage = document.getElementById('active-chart');
    const chartCaption = document.getElementById('chart-caption');
    const chartDownload = document.getElementById('chart-download');

    chartButtons.forEach((button) => {
      button.addEventListener('click', () => {
        chartButtons.forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
        chartImage.classList.add('is-switching');
        window.setTimeout(() => {
          chartImage.src = button.dataset.src;
          chartImage.alt = button.dataset.alt;
          chartCaption.textContent = button.dataset.title;
          chartDownload.href = button.dataset.src;
          chartImage.classList.remove('is-switching');
        }, 120);
      });
    });

    const videos = [...document.querySelectorAll('video')];
    videos.forEach((video) => video.addEventListener('play', () => {
      videos.filter((other) => other !== video).forEach((other) => other.pause());
    }));
  </script>
</body>
</html>
"""

    replacements = {
        "%%FRAME_RANGE%%": f"{START_FRAME}–{END_FRAME if END_FRAME else 'end'}",
        "%%BEST_RALLY%%": best_rally_text,
        "%%VALID_FRAMES%%": str(written_frames),
        "%%INVALID_FRAMES%%": str(invalid_frames),
        "%%BALL_DETECTIONS%%": str(ball_detection_frames),
        "%%DETECTION_RATE%%": f"{detection_rate:.1f}",
        "%%RALLY_COUNT%%": str(len(rallies)),
        "%%TOTAL_HITS%%": str(total_estimated_hits),
        "%%FASTEST_SPEED%%": f"{fastest_ball_speed:.0f}",
        "%%MOMENTUM_SWING%%": f"{momentum_swing:.1f}",
        "%%HIGHLIGHT_COUNT%%": str(highlight_count),
        "%%RALLY_ROWS%%": rally_rows,
        "%%INSPECTED_FRAMES%%": str(inspected_frames),
        "%%TIMELINE_DATA%%": timeline_json,
        "%%FPS%%": f"{fps:.6f}",
    }
    for placeholder, value in replacements.items():
        dashboard = dashboard.replace(placeholder, value)

    DASHBOARD_HTML_PATH.write_text(dashboard, encoding="utf-8")


def analyze_video() -> tuple[int, int, int, int, int, int, int, int, float, int, int]:
    """Create all annotated video, CSV, heatmap, score, and ball outputs."""
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
        roboflow_client = create_roboflow_client()
        roboflow_model_id = os.environ[ROBOFLOW_MODEL_ID_ENV]
        ball_prediction_cache = load_ball_prediction_cache()
        cache_prefix = f"{INPUT_VIDEO_PATH.name}:{roboflow_model_id}:"
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
        player_position_samples: list[PlayerPositionSample] = []
        valid_source_frames: list[int] = []
        last_score: Score | None = None
        recognized_score_samples = 0
        score_frame_numbers: list[int] = []
        player1_scores: list[int] = []
        player2_scores: list[int] = []
        ball_track_points: list[BallTrackPoint] = []
        ball_trail: deque[BallTrackPoint] = deque(maxlen=BALL_TRAIL_LENGTH)
        previous_ball_point: BallTrackPoint | None = None
        recent_raw_speeds: deque[float] = deque(
            maxlen=BALL_SPEED_SMOOTHING_WINDOW
        )
        ball_detection_frames = 0
        ball_inference_requests = 0
        ball_inference_failures = 0
        previous_video_frame: np.ndarray | None = None

        with (
            POSITIONS_CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file,
            BALL_POSITIONS_CSV_PATH.open(
                "w", newline="", encoding="utf-8"
            ) as ball_csv_file,
        ):
            csv_writer = csv.writer(csv_file)
            ball_csv_writer = csv.writer(ball_csv_file)
            csv_writer.writerow(
                [
                    "frame",
                    "player1_position_x",
                    "player1_position_y",
                    "player2_position_x",
                    "player2_position_y",
                ]
            )
            ball_csv_writer.writerow(
                [
                    "frame",
                    "ball_position_x",
                    "ball_position_y",
                    "confidence",
                    "speed_pixels_per_second",
                ]
            )

            while END_FRAME == 0 or frame_number <= END_FRAME:
                success, frame = capture.read()
                if not success:
                    break

                comparison_frame = previous_video_frame
                previous_video_frame = frame.copy()
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

                ball_inference_requests += 1
                try:
                    ball_detection = find_ball(
                        frame,
                        roboflow_client,
                        roboflow_model_id,
                        previous_frame=comparison_frame,
                        previous_point=previous_ball_point,
                        frame_number=current_frame_number,
                        prediction_cache=ball_prediction_cache,
                        cache_key=f"{cache_prefix}{current_frame_number}",
                    )
                except Exception as error:
                    ball_inference_failures += 1
                    if ball_inference_requests == 1:
                        raise RuntimeError(
                            "The first Roboflow inference request failed. Check the "
                            "API key, model ID, internet connection, and deployment status."
                        ) from None
                    if ball_inference_failures <= 3:
                        print(
                            f"Roboflow inference failed on frame "
                            f"{current_frame_number} ({type(error).__name__})."
                        )
                    ball_detection = None

                if ball_detection is not None:
                    raw_speed = calculate_ball_speed(
                        ball_detection,
                        current_frame_number,
                        previous_ball_point,
                        fps,
                    )
                    if raw_speed is None:
                        recent_raw_speeds.clear()
                        smoothed_speed = None
                    else:
                        recent_raw_speeds.append(raw_speed)
                        smoothed_speed = float(np.median(recent_raw_speeds))

                    if (
                        previous_ball_point is not None
                        and current_frame_number - previous_ball_point.frame_number
                        > BALL_MAX_TRACKING_GAP_FRAMES
                    ):
                        ball_trail.clear()

                    current_ball_point = BallTrackPoint(
                        frame_number=current_frame_number,
                        x=ball_detection.x,
                        y=ball_detection.y,
                        speed=smoothed_speed,
                    )
                    ball_track_points.append(current_ball_point)
                    ball_trail.append(current_ball_point)
                    previous_ball_point = current_ball_point
                    ball_detection_frames += 1
                    ball_csv_writer.writerow(
                        [
                            current_frame_number,
                            round(ball_detection.x, 2),
                            round(ball_detection.y, 2),
                            round(ball_detection.confidence, 4),
                            ""
                            if smoothed_speed is None
                            else round(smoothed_speed, 2),
                        ]
                    )
                else:
                    if (
                        previous_ball_point is not None
                        and current_frame_number - previous_ball_point.frame_number
                        > BALL_MAX_TRACKING_GAP_FRAMES
                    ):
                        ball_trail.clear()
                        recent_raw_speeds.clear()
                    ball_csv_writer.writerow([current_frame_number, "", "", "", ""])

                player1, player2 = find_player_positions(result)

                if player1 is None or player2 is None:
                    missing_position_frames += 1
                else:
                    player1_positions.append(player1)
                    player2_positions.append(player2)
                    player_position_samples.append(
                        PlayerPositionSample(
                            frame_number=current_frame_number,
                            player1=player1,
                            player2=player2,
                        )
                    )

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
                draw_ball_tracking(annotated_frame, ball_detection, ball_trail)
                writer.write(annotated_frame)
                valid_source_frames.append(current_frame_number)

                written_frames += 1

                if inspected_frames % 25 == 0:
                    save_ball_prediction_cache(ball_prediction_cache)
                    print(
                        f"Inspected {inspected_frames} frame(s); "
                        f"kept {written_frames}."
                    )

        if background_frame is None:
            raise RuntimeError("No video frames were processed.")

        writer.release()
        writer = None

        rallies = detect_rallies(ball_track_points, player_position_samples, fps)
        save_rally_statistics(rallies)
        create_rally_analysis_chart(rallies)

        momentum_samples = calculate_match_momentum(
            player_position_samples,
            ball_track_points,
            rallies,
            score_frame_numbers,
            player1_scores,
            player2_scores,
            fps,
        )
        save_match_momentum(momentum_samples)
        create_momentum_chart(momentum_samples, rallies)

        apply_rally_overlay(
            OUTPUT_VIDEO_PATH,
            valid_source_frames,
            rallies,
            fps,
            (width, height),
        )
        highlight_count = create_highlight_video(
            OUTPUT_VIDEO_PATH,
            valid_source_frames,
            rallies,
            fps,
            (width, height),
        )

        create_heatmap(background_frame, player1_positions, player2_positions)
        create_score_chart(score_frame_numbers, player1_scores, player2_scores)
        create_ball_speed_chart(ball_track_points)
        save_ball_prediction_cache(ball_prediction_cache)
        measured_speeds = [
            float(point.speed)
            for point in ball_track_points
            if point.speed is not None
        ]
        create_html_dashboard(
            inspected_frames,
            written_frames,
            invalid_frames,
            ball_detection_frames,
            max(measured_speeds, default=0.0),
            rallies,
            highlight_count,
            momentum_samples,
            ball_track_points,
            player_position_samples,
            valid_source_frames,
            score_frame_numbers,
            player1_scores,
            player2_scores,
            fps,
        )
        return (
            inspected_frames,
            written_frames,
            invalid_frames,
            len(player1_positions),
            missing_position_frames,
            recognized_score_samples,
            ball_detection_frames,
            ball_inference_failures,
            max(measured_speeds, default=0.0),
            len(rallies),
            highlight_count,
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
        ball_detection_frames,
        ball_inference_failures,
        fastest_ball_speed,
        rally_count,
        highlight_count,
    ) = analyze_video()

    if written_frames == 0:
        raise RuntimeError("No video frames were processed.")

    print(f"Finished inspecting {inspected_frames} frame(s).")
    print(f"Kept {written_frames} valid frame(s); skipped {invalid_frames} invalid frame(s).")
    print(f"Recorded both player positions in {position_frames} frame(s).")
    if missing_position_frames:
        print(f"Missing two-player detections in {missing_position_frames} frame(s).")
    print(f"Recognized the scoreboard in {recognized_score_samples} OCR sample(s).")
    print(f"Detected the ball in {ball_detection_frames} valid frame(s).")
    if ball_inference_failures:
        print(f"Roboflow inference failed in {ball_inference_failures} frame(s).")
    if fastest_ball_speed > 0:
        print(f"Fastest measured ball speed: {fastest_ball_speed:.0f} px/s.")
    print(f"Detected {rally_count} rally/rallies; selected {highlight_count} highlight(s).")
    print(f"Saved annotated video to: {OUTPUT_VIDEO_PATH.resolve()}")
    print(f"Saved player positions to: {POSITIONS_CSV_PATH.resolve()}")
    print(f"Saved player heatmap to: {HEATMAP_PATH.resolve()}")
    print(f"Saved score chart to: {SCORE_CHART_PATH.resolve()}")
    print(f"Saved ball positions to: {BALL_POSITIONS_CSV_PATH.resolve()}")
    print(f"Saved ball speed chart to: {BALL_SPEED_CHART_PATH.resolve()}")
    print(f"Saved rally statistics to: {RALLY_STATISTICS_CSV_PATH.resolve()}")
    print(f"Saved rally analysis to: {RALLY_ANALYSIS_CHART_PATH.resolve()}")
    print(f"Saved highlight reel to: {HIGHLIGHTS_VIDEO_PATH.resolve()}")
    print(f"Saved momentum data to: {MOMENTUM_CSV_PATH.resolve()}")
    print(f"Saved momentum chart to: {MOMENTUM_CHART_PATH.resolve()}")
    print(f"Saved responsive dashboard to: {DASHBOARD_HTML_PATH.resolve()}")


if __name__ == "__main__":
    main()
