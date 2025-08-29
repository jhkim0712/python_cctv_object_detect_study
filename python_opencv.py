import cv2
from ultralytics import YOLO

def main():
    model = YOLO('yolo11n.pt')  #yolo11n
    rtsp_url = "http://210.99.70.120:1935/live/cctv001.stream/playlist.m3u8"
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from stream.")
            break
        results = model(frame, stream=True)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if model.names[class_id] in ['person', 'bicycle', 'car', 'motorcycle', 'bus']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    label = f"{model.names[class_id]} {confidence:.2f}"
                    if confidence < 0.5:
                        color = (0, 0, 255)  # Red
                    elif confidence < 0.8:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 255, 0)  # Green
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imshow('CCTV Stream with Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()