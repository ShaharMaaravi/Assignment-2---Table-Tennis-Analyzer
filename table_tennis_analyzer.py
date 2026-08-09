"""Detect people and their poses in a single image with YOLO26."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_NAME = "yolo26n-pose.pt"
DEFAULT_OUTPUT_PATH = Path("output_with_detections.jpg")


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Draw person bounding boxes and pose joints on an image."
    )
    parser.add_argument("image", type=Path, help="Path to the input image")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Annotated image path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.25,
        help="Minimum detection confidence from 0 to 1 (default: 0.25)",
    )
    return parser.parse_args()


def validate_args(image_path: Path, output_path: Path, confidence: float) -> None:
    """Validate paths and prediction settings before loading the model."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if image_path.resolve() == output_path.resolve():
        raise ValueError("The output path must be different from the input path.")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0 and 1.")


def analyze_image(
    image_path: Path,
    output_path: Path,
    confidence: float,
) -> int:
    """Run pose detection, save the annotated image, and return the person count."""
    model = YOLO(MODEL_NAME)
    result = model.predict(source=str(image_path), conf=confidence, verbose=False)[0]

    # A pose Results object contains matching person boxes and body keypoints.
    annotated_image = result.plot(
        boxes=True,
        labels=True,
        kpt_line=True,
        kpt_radius=5,
        line_width=2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated_image):
        raise OSError(f"Could not save the annotated image to: {output_path}")

    return len(result.boxes)


def main() -> None:
    """Run the command-line program."""
    args = parse_args()
    validate_args(args.image, args.output, args.confidence)
    person_count = analyze_image(args.image, args.output, args.confidence)

    print(f"Detected {person_count} person(s).")
    print(f"Saved annotated image to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
