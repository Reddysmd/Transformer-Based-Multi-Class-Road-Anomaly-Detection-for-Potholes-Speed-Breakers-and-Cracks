from flask import Flask, Response
from ultralytics import YOLO
import cv2

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "road_defect_best.pt"
VIDEO_PATH = "video.mp4"   # downloaded video path
PHONE_STREAM_URL = ""      # will be set by user input

# Load YOLO model
model = YOLO(MODEL_PATH)

app = Flask(__name__)

cap = None   # global capture object

# ==============================
# MODE SELECTION
# ==============================
print("===================================")
print("🚦 Road Defect Detection System")
print("1 → Mobile Live Stream (Phone Camera IP)")
print("2 → Downloaded Video")
print("===================================")

choice = input("Enter your choice (1 or 2): ").strip()

if choice == "1":
    print("\n📱 MOBILE CAMERA MODE SELECTED")
    print("Install app on phone: 👉 IP Webcam (Android)")
    print("Format example:")
    print("http://192.168.1.10:8080/video")
    PHONE_STREAM_URL = input("\nEnter phone camera IP stream URL: ").strip()

    cap = cv2.VideoCapture(PHONE_STREAM_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("❌ Failed to connect to phone camera stream")
        exit()
    else:
        print("✅ Connected to phone camera")

elif choice == "2":
    print("\n📁 DOWNLOADED VIDEO MODE SELECTED")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ Failed to load video file")
        exit()
    else:
        print("🎬 Video loaded:", VIDEO_PATH)

else:
    print("❌ Invalid choice. Exiting...")
    exit()

# ==============================
# STREAM GENERATOR
# ==============================
def generate_frames():
    global cap
    while True:
        if cap is None:
            continue

        success, frame = cap.read()

        if not success:
            # loop video if downloaded video
            if choice == "2":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                continue

        # YOLO Detection
        results = model(frame, conf=0.4)
        annotated_frame = results[0].plot()

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # MJPEG Stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ==============================
# FLASK ROUTES
# ==============================
@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Road Defect Detection</title>
        </head>
        <body style="text-align:center;background:#0f172a;color:white;">
            <h1>🚦 Road Defect Detection Live</h1>
            <img src="/video" width="85%">
        </body>
    </html>
    """

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    print("\n===================================")
    print("📡 Mobile Stream Server Started")
    print("👉 Open in phone browser using this IP:\n")

    print("http://<PC-IP>:5000")
    print("\nExample:")
    print("http://192.168.1.5:5000")

    print("===================================\n")

    app.run(host="0.0.0.0", port=5000, debug=False)