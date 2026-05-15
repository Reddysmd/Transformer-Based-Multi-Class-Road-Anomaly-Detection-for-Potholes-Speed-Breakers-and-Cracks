from ultralytics import YOLO
import cv2

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "road_defect_best.pt"
VIDEO_PATH = "video.mp4"

# Load YOLO model
model = YOLO(MODEL_PATH)

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
    print("Example IP: http://192.168.1.10:8080/video")
    PHONE_STREAM_URL = input("Enter phone camera IP stream URL: ").strip()
    cap = cv2.VideoCapture(PHONE_STREAM_URL)

elif choice == "2":
    print("\n📁 DOWNLOADED VIDEO MODE SELECTED")
    cap = cv2.VideoCapture(VIDEO_PATH)

else:
    print("❌ Invalid choice. Exiting...")
    exit()

# ==============================
# PROCESS & DISPLAY
# ==============================
while True:
    success, frame = cap.read()
    if not success:
        print("⚠️ End of video or failed to read frame")
        break

    # YOLO Detection
    results = model(frame, conf=0.4)
    annotated_frame = results[0].plot()

    # Show output on laptop screen
    cv2.imshow("🚦 Road Defect Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Stopped by user")
        break

# ==============================
# CLEANUP
# ==============================
cap.release()
cv2.destroyAllWindows()