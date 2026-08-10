"""Create a diverse, upload-ready Roboflow image set from the match video."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Importing the analyzer also imports matplotlib. Keep its cache outside the project.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

from table_tennis_analyzer import is_valid_frame  # noqa: E402


DEFAULT_INPUT = Path("input.mp4")
DEFAULT_OUTPUT = Path("roboflow_ball_dataset")
POSE_MODEL = "yolo26n-pose.pt"

CANDIDATE_COUNT = 720
TARGET_MATCH_FRAMES = 180
TARGET_CONTEXT_FRAMES = 30
TEMPORAL_BUCKETS = 24
MIN_SHARPNESS = 55.0
JPEG_QUALITY = 92


@dataclass
class Candidate:
    frame_number: int
    frame: np.ndarray
    valid_side_view: bool
    sharpness: float
    ball_candidate_score: float
    image_hash: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--match-frames", type=int, default=TARGET_MATCH_FRAMES)
    parser.add_argument("--context-frames", type=int, default=TARGET_CONTEXT_FRAMES)
    parser.add_argument("--candidates", type=int, default=CANDIDATE_COUNT)
    return parser.parse_args()


def read_frame_triplet(
    capture: cv2.VideoCapture, frame_number: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Read the preceding, selected, and following frame."""
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - 1))
    ok_previous, previous = capture.read()
    ok_current, current = capture.read()
    ok_next, following = capture.read()
    if not (ok_previous and ok_current and ok_next):
        return None
    return previous, current, following


def frame_sharpness(frame: np.ndarray) -> float:
    """Return a scale-independent Laplacian sharpness estimate."""
    height, width = frame.shape[:2]
    scale = min(1.0, 960 / width)
    resized = cv2.resize(
        frame,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def difference_hash(frame: np.ndarray) -> np.ndarray:
    """Create a compact hash used to reject near-duplicate broadcast frames."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tiny = cv2.resize(gray, (17, 16), interpolation=cv2.INTER_AREA)
    return (tiny[:, 1:] > tiny[:, :-1]).reshape(-1)


def likely_ball_score(
    previous: np.ndarray, current: np.ndarray, following: np.ndarray
) -> float:
    """Score small bright moving blobs that may be a table-tennis ball.

    This is only a ranking heuristic. It deliberately does not create annotations;
    every saved image still needs to be reviewed in Roboflow.
    """
    height, width = current.shape[:2]
    x1, x2 = int(width * 0.06), int(width * 0.94)
    y1, y2 = int(height * 0.16), int(height * 0.82)

    before = previous[y1:y2, x1:x2]
    image = current[y1:y2, x1:x2]
    after = following[y1:y2, x1:x2]

    motion_before = cv2.cvtColor(cv2.absdiff(image, before), cv2.COLOR_BGR2GRAY)
    motion_after = cv2.cvtColor(cv2.absdiff(image, after), cv2.COLOR_BGR2GRAY)
    motion = cv2.max(motion_before, motion_after)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    white = (saturation < 95) & (value > 145)
    orange = (hue >= 3) & (hue <= 28) & (saturation > 75) & (value > 120)
    mask = ((white | orange) & (motion > 24)).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scores: list[float] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, blob_width, blob_height = cv2.boundingRect(contour)
        if not 1.0 <= area <= 150.0:
            continue
        if not (1 <= blob_width <= 22 and 1 <= blob_height <= 22):
            continue
        aspect = min(blob_width, blob_height) / max(blob_width, blob_height)
        if aspect < 0.28:
            continue

        blob_motion = float(motion[y : y + blob_height, x : x + blob_width].mean())
        scores.append(blob_motion * aspect / (1.0 + abs(area - 18.0) / 30.0))

    return max(scores, default=0.0)


def is_near_duplicate(candidate: Candidate, selected: list[Candidate]) -> bool:
    return any(
        int(np.count_nonzero(candidate.image_hash != item.image_hash)) < 9
        for item in selected
    )


def balanced_selection(
    candidates: list[Candidate], target: int, total_frames: int
) -> list[Candidate]:
    """Select high-scoring frames while preserving coverage across the video."""
    buckets: list[list[Candidate]] = [[] for _ in range(TEMPORAL_BUCKETS)]
    for candidate in candidates:
        bucket = min(
            TEMPORAL_BUCKETS - 1,
            candidate.frame_number * TEMPORAL_BUCKETS // max(1, total_frames),
        )
        buckets[bucket].append(candidate)

    for bucket in buckets:
        bucket.sort(
            key=lambda item: (item.ball_candidate_score, item.sharpness), reverse=True
        )

    selected: list[Candidate] = []
    rank = 0
    while len(selected) < target:
        added = False
        for bucket in buckets:
            if rank >= len(bucket):
                continue
            candidate = bucket[rank]
            if not is_near_duplicate(candidate, selected):
                selected.append(candidate)
                added = True
                if len(selected) == target:
                    break
        rank += 1
        if not added and all(rank >= len(bucket) for bucket in buckets):
            break

    return sorted(selected, key=lambda item: item.frame_number)


def write_contact_sheets(
    selected: list[tuple[str, Candidate]], output_directory: Path
) -> None:
    sheet_directory = output_directory / "contact_sheets_do_not_upload"
    sheet_directory.mkdir(parents=True, exist_ok=True)
    columns, rows = 5, 5
    thumb_width, thumb_height = 384, 216

    for page_start in range(0, len(selected), columns * rows):
        page = selected[page_start : page_start + columns * rows]
        canvas = np.full(
            (rows * (thumb_height + 30), columns * thumb_width, 3), 245, dtype=np.uint8
        )
        for index, (filename, candidate) in enumerate(page):
            row, column = divmod(index, columns)
            thumbnail = cv2.resize(
                candidate.frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA
            )
            top, left = row * (thumb_height + 30), column * thumb_width
            canvas[top : top + thumb_height, left : left + thumb_width] = thumbnail
            cv2.putText(
                canvas,
                filename,
                (left + 6, top + thumb_height + 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

        page_number = page_start // (columns * rows) + 1
        cv2.imwrite(str(sheet_directory / f"contact_sheet_{page_number:02d}.jpg"), canvas)


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Video not found: {args.input}")

    images_directory = args.output / "images_to_upload"
    images_directory.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {args.input}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if total_frames < 3 or fps <= 0:
        raise ValueError("The input video has invalid frame metadata.")

    frame_numbers = np.linspace(
        1, total_frames - 2, min(args.candidates, total_frames - 2), dtype=int
    )
    model = YOLO(POSE_MODEL)
    candidates: list[Candidate] = []
    batch_frames: list[np.ndarray] = []
    batch_metadata: list[tuple[int, float, float, np.ndarray]] = []

    def analyze_batch() -> None:
        if not batch_frames:
            return
        results = model(batch_frames, conf=0.25, imgsz=640, verbose=False)
        for frame, result, metadata in zip(batch_frames, results, batch_metadata):
            frame_number, sharpness, ball_score, image_hash = metadata
            candidates.append(
                Candidate(
                    frame_number=frame_number,
                    frame=frame,
                    valid_side_view=is_valid_frame(frame, result),
                    sharpness=sharpness,
                    ball_candidate_score=ball_score,
                    image_hash=image_hash,
                )
            )
        batch_frames.clear()
        batch_metadata.clear()

    for position, frame_number in enumerate(frame_numbers, start=1):
        triplet = read_frame_triplet(capture, int(frame_number))
        if triplet is None:
            continue
        previous, current, following = triplet
        sharpness = frame_sharpness(current)
        if sharpness < MIN_SHARPNESS:
            continue

        batch_frames.append(current)
        batch_metadata.append(
            (
                int(frame_number),
                sharpness,
                likely_ball_score(previous, current, following),
                difference_hash(current),
            )
        )
        if len(batch_frames) == 24:
            analyze_batch()
        if position % 120 == 0:
            print(f"Scanned {position}/{len(frame_numbers)} candidate frames...")

    analyze_batch()
    capture.release()

    match_candidates = [item for item in candidates if item.valid_side_view]
    context_candidates = [
        item
        for item in candidates
        if not item.valid_side_view and item.ball_candidate_score == 0.0
    ]
    match_frames = balanced_selection(match_candidates, args.match_frames, total_frames)
    context_frames = balanced_selection(
        context_candidates, args.context_frames, total_frames
    )

    selected: list[tuple[str, Candidate]] = []
    for category, items in (("match", match_frames), ("context", context_frames)):
        for index, candidate in enumerate(items, start=1):
            filename = (
                f"{category}_{index:03d}_source_frame_{candidate.frame_number:06d}.jpg"
            )
            destination = images_directory / filename
            cv2.imwrite(
                str(destination),
                candidate.frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            selected.append((filename, candidate))

    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "file_name",
                "source_frame",
                "time_seconds",
                "category",
                "valid_side_view",
                "ball_candidate_score",
                "sharpness",
            ]
        )
        for filename, candidate in selected:
            writer.writerow(
                [
                    filename,
                    candidate.frame_number,
                    round(candidate.frame_number / fps, 2),
                    filename.split("_", 1)[0],
                    candidate.valid_side_view,
                    round(candidate.ball_candidate_score, 2),
                    round(candidate.sharpness, 2),
                ]
            )

    guide = (
        "ROBOFLOW UPLOAD GUIDE\n"
        "=====================\n"
        "1. Create an Object Detection project with class: table-tennis-ball\n"
        "2. Upload only the files inside images_to_upload/.\n"
        "3. Draw a tight box around every visible ball, including small/blurred balls.\n"
        "4. Mark an image as null/background only after confirming no ball is visible.\n"
        "5. Do not upload manifest.csv or contact_sheets_do_not_upload/.\n\n"
        "The small-moving-blob score is only for dataset selection; it is not an annotation.\n"
    )
    (args.output / "README_UPLOAD.txt").write_text(guide, encoding="utf-8")
    write_contact_sheets(selected, args.output)

    print(
        f"Created {len(match_frames)} match frames and {len(context_frames)} context "
        f"frames in {images_directory}"
    )


if __name__ == "__main__":
    main()
