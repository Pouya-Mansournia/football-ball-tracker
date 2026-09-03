"""
V1 - Ball Detection & Trajectory
--------------------------------
Reads a football video frame by frame, runs a pretrained YOLO model to detect
the "sports ball" class, computes the ball center per frame, keeps a rolling
trajectory, and writes an annotated output video.

Usage:
    python src/detect_ball.py --video videos/football_match_01.mp4 --output output/ball_tracked.mp4
"""

import argparse
from collections import deque

import cv2
from ultralytics import YOLO

# COCO class id for "sports ball"
BALL_CLASS_ID = 32

TRAJECTORY_LEN = 40  # how many past ball centers to draw


def parse_args():
    parser = argparse.ArgumentParser(description="Detect and track the ball in a football video.")
    parser.add_argument("--video", default="videos/football_match_01.mp4", help="Path to input video.")
    parser.add_argument("--output", default="output/ball_tracked.mp4", help="Path to output video.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO model to use.")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold for the ball class.")
    parser.add_argument("--show", action="store_true", help="Show a live preview window while processing.")
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    trajectory = deque(maxlen=TRAJECTORY_LEN)

    frame_idx = 0
    detected_count = 0

    print(f"Processing {args.video} ({width}x{height} @ {fps:.1f}fps)...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            frame,
            classes=[BALL_CLASS_ID],
            conf=args.conf,
            verbose=False,
        )[0]

        ball_center = None
        best_conf = -1.0

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                ball_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                best_box = (int(x1), int(y1), int(x2), int(y2))

        if ball_center is not None:
            detected_count += 1
            trajectory.append(ball_center)
            x1, y1, x2, y2 = best_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, ball_center, 3, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"ball {best_conf:.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            print(f"Frame {frame_idx} -> Ball {ball_center}")

        # Draw trajectory trail
        for i in range(1, len(trajectory)):
            if trajectory[i - 1] is None or trajectory[i] is None:
                continue
            cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 0, 0), 2)

        writer.write(frame)

        if args.show:
            cv2.imshow("Ball Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Done. {detected_count}/{frame_idx} frames had a detected ball.")
    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
