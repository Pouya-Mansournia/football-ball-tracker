"""
V1.5 - Ball Detection + Kalman Tracking
----------------------------------------
Same pipeline as detect_ball.py, but bridges the gaps between YOLO detections
with a Kalman filter: while the ball is detected, the filter is corrected with
the measurement; when detection is missed for a short while, the filter's
prediction is used instead so the trajectory stays continuous.

Usage:
    python src/track_ball.py --video videos/football_match_01.mp4 --output output/ball_tracked_kalman.mp4
"""

import argparse
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

BALL_CLASS_ID = 32  # COCO "sports ball"
TRAJECTORY_LEN = 40
MAX_MISSED_FRAMES = 15  # stop trusting predictions after this many misses in a row
MAX_JUMP_FRACTION = 0.06  # a detection farther than this fraction of the frame
                           # width from the last known position is treated as a
                           # false positive, not the ball (scales with resolution)


def parse_args():
    parser = argparse.ArgumentParser(description="Track the ball with YOLO detection + Kalman prediction.")
    parser.add_argument("--video", default="videos/football_match_01.mp4", help="Path to input video.")
    parser.add_argument("--output", default="output/ball_tracked_kalman.mp4", help="Path to output video.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO model to use.")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold for the ball class.")
    parser.add_argument("--max-box-area-ratio", type=float, default=0.03,
                         help="Reject ball boxes larger than this fraction of the frame area "
                              "(raise for close-up footage, lower for wide broadcast shots).")
    parser.add_argument("--max-aspect-ratio", type=float, default=1.6,
                         help="Reject ball boxes more elongated than this width/height ratio.")
    parser.add_argument("--show", action="store_true", help="Show a live preview window while processing.")
    return parser.parse_args()


def make_kalman():
    """Constant-velocity Kalman filter over (x, y, vx, vy)."""
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    return kf


def best_ball_box(results, frame_area, max_box_area_ratio=0.03, max_aspect_ratio=1.6):
    """Return (x1, y1, x2, y2, conf) for the highest-confidence *plausible* ball box.

    The real ball in a broadcast shot is small and roughly square. Filtering
    out large / elongated boxes rejects a lot of false positives (heads,
    shoulders, other round-ish objects) that "sports ball" sometimes fires on.
    """
    best = None
    best_conf = -1.0
    for box in results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue

        area_ratio = (w * h) / frame_area
        aspect_ratio = max(w, h) / min(w, h)

        if area_ratio > max_box_area_ratio or aspect_ratio > max_aspect_ratio:
            continue  # too big or too elongated to be the ball

        if conf > best_conf:
            best_conf = conf
            best = (int(x1), int(y1), int(x2), int(y2), conf)
    return best


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

    kalman = make_kalman()
    kalman_initialized = False
    missed_in_a_row = 0
    last_ball_center = None

    trajectory = deque(maxlen=TRAJECTORY_LEN)

    frame_idx = 0
    detected_count = 0
    predicted_count = 0

    print(f"Processing {args.video} ({width}x{height} @ {fps:.1f}fps)...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, classes=[BALL_CLASS_ID], conf=args.conf, verbose=False)[0]
        detection = best_ball_box(
            results,
            frame_area=width * height,
            max_box_area_ratio=args.max_box_area_ratio,
            max_aspect_ratio=args.max_aspect_ratio,
        )

        ball_center = None
        source = None  # "detected" or "predicted"

        if detection is not None and last_ball_center is not None:
            x1, y1, x2, y2, conf = detection
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            jump = ((cx - last_ball_center[0]) ** 2 + (cy - last_ball_center[1]) ** 2) ** 0.5
            allowed_jump = (MAX_JUMP_FRACTION * width) * (1 + missed_in_a_row)
            if jump > allowed_jump:
                detection = None  # likely a false positive far from the last known ball position

        if detection is not None:
            x1, y1, x2, y2, conf = detection
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if not kalman_initialized:
                kalman.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                kalman.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                kalman_initialized = True

            kalman.predict()
            corrected = kalman.correct(np.array([[cx], [cy]], dtype=np.float32))
            ball_center = (int(corrected[0, 0]), int(corrected[1, 0]))
            source = "detected"
            missed_in_a_row = 0
            detected_count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ball {conf:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        elif kalman_initialized and missed_in_a_row < MAX_MISSED_FRAMES:
            predicted = kalman.predict()
            px, py = int(predicted[0, 0]), int(predicted[1, 0])
            in_bounds = 0 <= px < width and 0 <= py < height

            if in_bounds:
                ball_center = (px, py)
                source = "predicted"
                missed_in_a_row += 1
                predicted_count += 1

                cv2.circle(frame, ball_center, 12, (0, 165, 255), 2)
                cv2.putText(frame, "predicted", (ball_center[0] - 20, ball_center[1] - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
            else:
                # Prediction has drifted off-screen (likely from a bad detection
                # earlier) - stop trusting it until the ball is detected again.
                kalman_initialized = False
                missed_in_a_row += 1
        else:
            missed_in_a_row += 1

        if ball_center is not None:
            last_ball_center = ball_center
            trajectory.append(ball_center)
            color = (0, 0, 255) if source == "detected" else (0, 165, 255)
            cv2.circle(frame, ball_center, 3, color, -1)
            print(f"Frame {frame_idx} -> Ball {ball_center} ({source})")

        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 0, 0), 2)

        writer.write(frame)

        if args.show:
            cv2.imshow("Ball Tracking (Kalman)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    total_with_ball = detected_count + predicted_count
    print(
        f"Done. {detected_count} detected + {predicted_count} predicted "
        f"= {total_with_ball}/{frame_idx} frames with a ball position "
        f"({100 * total_with_ball / frame_idx:.1f}%)."
    )
    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
