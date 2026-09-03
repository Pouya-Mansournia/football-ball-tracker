"""
V2 - Player Detection & Tracking (+ ball)
------------------------------------------
Extends the V1.5 ball tracker with player detection and tracking. Uses
Ultralytics' built-in ByteTrack integration (model.track with persist=True) so
each player keeps a stable ID across frames, while the ball keeps going
through the Kalman-filter + gating pipeline from track_ball.py.

Usage:
    python src/track_match.py --video videos/football_match_02.mp4 --output output/match_tracked.mp4
"""

import argparse
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32

TRAJECTORY_LEN = 40
MAX_MISSED_FRAMES = 15
MAX_JUMP_FRACTION = 0.06

# Distinct BGR colors cycled by player track ID so each player is visually
# consistent across frames without needing per-team color detection yet.
PLAYER_COLORS = [
    (255, 99, 71), (60, 179, 113), (255, 215, 0), (186, 85, 211),
    (64, 224, 208), (255, 140, 0), (30, 144, 255), (240, 128, 128),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Track players and the ball in a football video.")
    parser.add_argument("--video", default="videos/football_match_02.mp4", help="Path to input video.")
    parser.add_argument("--output", default="output/match_tracked.mp4", help="Path to output video.")
    parser.add_argument("--model", default="yolov8s.pt", help="Ultralytics YOLO model to use.")
    parser.add_argument("--ball-conf", type=float, default=0.15, help="Confidence threshold for the ball class.")
    parser.add_argument("--player-conf", type=float, default=0.3, help="Confidence threshold for the person class.")
    parser.add_argument("--max-box-area-ratio", type=float, default=0.03,
                         help="Reject ball boxes larger than this fraction of the frame area.")
    parser.add_argument("--max-aspect-ratio", type=float, default=1.6,
                         help="Reject ball boxes more elongated than this width/height ratio.")
    parser.add_argument("--tracker", default="configs/bytetrack_stable.yaml",
                         help="ByteTrack config to use for player ID tracking.")
    parser.add_argument("--show", action="store_true", help="Show a live preview window while processing.")
    return parser.parse_args()


def make_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    return kf


def best_ball_box(boxes, frame_area, max_box_area_ratio, max_aspect_ratio):
    """boxes: iterable of (x1, y1, x2, y2, conf) already filtered to the ball class."""
    best = None
    best_conf = -1.0
    for x1, y1, x2, y2, conf in boxes:
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        area_ratio = (w * h) / frame_area
        aspect_ratio = max(w, h) / min(w, h)
        if area_ratio > max_box_area_ratio or aspect_ratio > max_aspect_ratio:
            continue
        if conf > best_conf:
            best_conf = conf
            best = (int(x1), int(y1), int(x2), int(y2), conf)
    return best


def player_color(track_id):
    if track_id is None:
        return (200, 200, 200)
    return PLAYER_COLORS[int(track_id) % len(PLAYER_COLORS)]


def main():
    args = parse_args()
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_area = width * height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    kalman = make_kalman()
    kalman_initialized = False
    missed_in_a_row = 0
    last_ball_center = None
    trajectory = deque(maxlen=TRAJECTORY_LEN)

    frame_idx = 0
    ball_detected_count = 0
    ball_predicted_count = 0
    seen_player_ids = set()

    print(f"Processing {args.video} ({width}x{height} @ {fps:.1f}fps)...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(
            frame,
            classes=[PERSON_CLASS_ID, BALL_CLASS_ID],
            conf=min(args.ball_conf, args.player_conf),
            persist=True,
            tracker=args.tracker,
            verbose=False,
        )[0]

        ball_candidates = []
        player_boxes = []  # (x1, y1, x2, y2, conf, track_id)

        if results.boxes is not None:
            track_ids = results.boxes.id
            for i, box in enumerate(results.boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                track_id = int(track_ids[i]) if track_ids is not None else None

                if cls == BALL_CLASS_ID and conf >= args.ball_conf:
                    ball_candidates.append((x1, y1, x2, y2, conf))
                elif cls == PERSON_CLASS_ID and conf >= args.player_conf:
                    player_boxes.append((x1, y1, x2, y2, conf, track_id))

        # --- Draw players ---
        for x1, y1, x2, y2, conf, track_id in player_boxes:
            color = player_color(track_id)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"#{track_id}" if track_id is not None else "player"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if track_id is not None:
                seen_player_ids.add(track_id)

        # --- Ball detection + Kalman gating (same approach as track_ball.py) ---
        detection = best_ball_box(ball_candidates, frame_area, args.max_box_area_ratio, args.max_aspect_ratio)

        if detection is not None and last_ball_center is not None:
            x1, y1, x2, y2, conf = detection
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            jump = ((cx - last_ball_center[0]) ** 2 + (cy - last_ball_center[1]) ** 2) ** 0.5
            allowed_jump = (MAX_JUMP_FRACTION * width) * (1 + missed_in_a_row)
            if jump > allowed_jump:
                detection = None

        ball_center = None
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
            missed_in_a_row = 0
            ball_detected_count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ball {conf:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        elif kalman_initialized and missed_in_a_row < MAX_MISSED_FRAMES:
            predicted = kalman.predict()
            px, py = int(predicted[0, 0]), int(predicted[1, 0])
            if 0 <= px < width and 0 <= py < height:
                ball_center = (px, py)
                missed_in_a_row += 1
                ball_predicted_count += 1
                cv2.circle(frame, ball_center, 12, (0, 165, 255), 2)
                cv2.putText(frame, "predicted", (ball_center[0] - 20, ball_center[1] - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
            else:
                kalman_initialized = False
                missed_in_a_row += 1
        else:
            missed_in_a_row += 1

        if ball_center is not None:
            last_ball_center = ball_center
            trajectory.append(ball_center)

        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 0, 0), 2)

        # --- Nearest player to ball (possession proxy) ---
        if ball_center is not None and player_boxes:
            def dist_to_ball(p):
                x1, y1, x2, y2, conf, track_id = p
                pcx, pcy = (x1 + x2) / 2, (y1 + y2) / 2
                return (pcx - ball_center[0]) ** 2 + (pcy - ball_center[1]) ** 2

            nearest = min(player_boxes, key=dist_to_ball)
            nx1, ny1, nx2, ny2, nconf, nid = nearest
            cv2.putText(frame, "possession?", (int(nx1), int(ny2) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        writer.write(frame)

        if args.show:
            cv2.imshow("Match Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    total_ball_coverage = ball_detected_count + ball_predicted_count
    print(
        f"Done. {frame_idx} frames processed. "
        f"Ball: {ball_detected_count} detected + {ball_predicted_count} predicted "
        f"= {total_ball_coverage}/{frame_idx} ({100 * total_ball_coverage / frame_idx:.1f}%). "
        f"Unique player IDs seen: {len(seen_player_ids)}."
    )
    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
