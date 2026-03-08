"""
Live stream router:
  GET  /api/video/feed  — MJPEG stream (embed as <img src="...">)
  WS   /ws/stream       — JSON push of latest psychological state
"""

import asyncio
import cv2

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from api.session_manager import session_manager

router = APIRouter()


# ------------------------------------------------------------------ #
# MJPEG video stream                                                   #
# ------------------------------------------------------------------ #

async def _mjpeg_generator():
    """Yield MJPEG frames from the background camera loop."""
    while True:
        frame = session_manager.get_latest_frame()
        if frame is not None:
            _, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
        await asyncio.sleep(0.033)  # ~30 fps


@router.get("/api/video/feed")
async def video_feed():
    """MJPEG stream endpoint — usable as <img src="/api/video/feed">."""
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ------------------------------------------------------------------ #
# WebSocket real-time state stream                                      #
# ------------------------------------------------------------------ #

@router.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    """Push latest psychological state to the browser every 100 ms."""
    await ws.accept()
    try:
        while True:
            state = session_manager.get_latest_state()
            if state:
                await ws.send_json(state)
            else:
                # Send a heartbeat so the client knows we're alive
                await ws.send_json({"status": "waiting"})
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
