from flask import Flask, Response
import cv2

app = Flask(__name__)

print("===================================")
print("📱 PHONE CAMERA STREAM TEST")
print("Use IP Webcam app on phone")
print("Format: http://<PHONE-IP>:8080/video")
print("Example: http://192.168.1.10:8080/video")
print("===================================")

PHONE_STREAM_URL = input("Enter phone camera stream URL: ").strip()

cap = cv2.VideoCapture(PHONE_STREAM_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ ERROR: Cannot connect to phone camera stream")
    print("Check IP, WiFi, and IP Webcam app")
    exit()
else:
    print("✅ Connected to phone camera stream")

def generate_frames():
    while True:
        success, frame = cap.read()

        if not success or frame is None:
            print("⚠ Frame not received")
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return """
    <html>
        <head><title>Phone Camera Test</title></head>
        <body style="text-align:center;background:black;color:white;">
            <h2>📱 Phone Camera Live Test</h2>
            <img src="/video" width="90%">
        </body>
    </html>
    """

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("\n===================================")
    print("📡 Test Server Started")
    print("👉 Open on phone browser:\n")
    print("http://<PC-IP>:5000")
    print("\nExample:")
    print("http://10.50.49.116:5000")
    print("===================================\n")

    app.run(host="0.0.0.0", port=5000, debug=False)