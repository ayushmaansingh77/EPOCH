# yolo_ws_server.py
# Minimal WebSocket server that streams YOLO detections + JPEG frames (base64).
# Fixed: handler accepts either (ws, path) or (ws,) for compatibility with websockets versions.

import argparse
import asyncio
import base64
import json
import time
from typing import List

import cv2
import numpy as np

# required libs
try:
    import websockets
except Exception:
    raise SystemExit("Please install websockets: pip install websockets")

try:
    from ultralytics import YOLO
except Exception:
    raise SystemExit("Please install ultralytics: pip install ultralytics")

# Optional filter of COCO vehicle classes (comment out to send all)
VEHICLE_CLASS_IDS = {2, 3, 5, 7}


def normalize_detections_from_results(results, frame_w: int, frame_h: int) -> List[dict]:
    dets = []
    if len(results) == 0:
        return dets
    res = results[0]
    if not hasattr(res, "boxes"):
        return dets
    for box in res.boxes:
        try:
            xy = box.xyxy[0].tolist()
        except Exception:
            try:
                xy = box.xyxy.tolist()[0]
            except Exception:
                continue
        x1, y1, x2, y2 = xy
        try:
            cls = int(box.cls[0])
        except Exception:
            cls = int(box.cls)
        try:
            conf = float(box.conf[0])
        except Exception:
            conf = float(box.conf)
        if VEHICLE_CLASS_IDS and cls not in VEHICLE_CLASS_IDS:
            continue
        dets.append({
            "cls": cls,
            "x1": x1 / frame_w,
            "y1": y1 / frame_h,
            "x2": x2 / frame_w,
            "y2": y2 / frame_h,
            "conf": conf,
            "norm": True,
        })
    return dets


async def stream_loop(ws, model, cap, target_fps=10, frame_max_side=640, jpeg_q=60):
    interval = 1.0 / max(1.0, target_fps)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of source or cannot read frame.")
                break
            h, w = frame.shape[:2]

            # inference
            try:
                results = model(frame, verbose=False)
            except Exception as e:
                print("YOLO inference error:", e)
                results = []

            detections = normalize_detections_from_results(results, w, h)

            # resize for bandwidth
            max_side = max(w, h)
            if frame_max_side and max_side > frame_max_side:
                scale = frame_max_side / float(max_side)
                new_w, new_h = int(w * scale), int(h * scale)
                small = cv2.resize(frame, (new_w, new_h))
            else:
                small = frame

            ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)])
            jpeg_b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None

            payload = {
                "detections": detections,
                "frame_jpeg_b64": jpeg_b64,
                "timestamp": int(time.time() * 1000),
            }

            try:
                await ws.send(json.dumps(payload))
            except Exception as e:
                # connection closed or send failed
                print("Send failed:", e)
                break

            await asyncio.sleep(interval)
    finally:
        try:
            cap.release()
        except Exception:
            pass
        print("Stream loop ended.")


async def handler(ws, path, args, model):
    # actual handler that expects websocket and path
    print("Client connected (handler).")
    if args.mode == "cam":
        cap = cv2.VideoCapture(args.cam)
    else:
        cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print("Failed to open capture:", args.video if args.mode == "file" else f"cam:{args.cam}")
        try:
            await ws.close()
        except Exception:
            pass
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    await stream_loop(ws, model, cap, target_fps=min(args.fps, max(1, int(fps))),
                      frame_max_side=args.frame_scale, jpeg_q=args.jpeg_quality)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["file", "cam"], default="file")
    p.add_argument("--video", default="traffic.mp4")
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--fps", type=float, default=8.0, help="target FPS to send to clients")
    p.add_argument("--frame-scale", type=int, default=640, help="max side of frame to scale to (reduce bandwidth)")
    p.add_argument("--jpeg-quality", type=int, default=60, help="JPEG quality (0-100)")
    return p.parse_args()


async def main_async(args):
    # load model
    try:
        model = YOLO(args.model)
    except Exception as e:
        raise SystemExit("Failed to load YOLO model: " + str(e))
    print("Loaded YOLO model:", args.model)

    # wrapper handler accepts either (ws, path) or (ws,)
    async def ws_handler(ws, path=None):
        try:
            await handler(ws, path, args, model)
        except Exception as ex:
            print("connection handler failed")
            import traceback
            traceback.print_exc()

    server = await websockets.serve(ws_handler, "0.0.0.0", args.port)
    print(f"WebSocket server listening on ws://0.0.0.0:{args.port} (source={args.mode})")
    try:
        await asyncio.Future()  # run forever
    finally:
        server.close()
        await server.wait_closed()


def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Server stopped by user")


if __name__ == "__main__":
    main()
