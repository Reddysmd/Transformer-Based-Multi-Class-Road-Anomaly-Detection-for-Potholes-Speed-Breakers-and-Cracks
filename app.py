from flask import Flask, Response, render_template
from ultralytics import YOLO
import cv2

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "road_defect_best.pt"
VIDEO_PATH = "video.mp4"

model = YOLO(MODEL_PATH)
app = Flask(__name__)

cap = None
mode = None   # "phone" or "video"

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
    mode = "phone"
    print("\n📱 MOBILE CAMERA MODE SELECTED")
    print("Format: http://<PHONE-IP>:8080/video")
    PHONE_STREAM_URL = input("Enter phone camera IP stream URL: ").strip()
    cap = cv2.VideoCapture(PHONE_STREAM_URL, cv2.CAP_FFMPEG)

elif choice == "2":
    mode = "video"
    print("\n📁 DOWNLOADED VIDEO MODE SELECTED")
    cap = cv2.VideoCapture(VIDEO_PATH)

else:
    print("❌ Invalid choice")
    exit()

if not cap.isOpened():
    print("❌ Camera/Video not opened")
    exit()

# ==============================
# STREAM GENERATOR
# ==============================
def generate_frames():
    global cap, mode
    while True:
        success, frame = cap.read()

        if not success:
            if mode == "video":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                continue

        # Resize for better detection
        frame = cv2.resize(frame, (640, 640))

        # YOLO Detection
        results = model.predict(frame, conf=0.15, verbose=False)
        annotated_frame = results[0].plot()

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ==============================
# ROUTES
# ==============================
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    print("\n===================================")
    print("📡 Server Started")
    print("👉 Open in browser:")
    print("http://<PC-IP>:5000")
    print("===================================\n")

    app.run(host="0.0.0.0", port=5000, debug=False)