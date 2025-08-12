from ultralytics import YOLO
import cv2

model = YOLO('yolo11n.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        results = model(frame,verbose= False)
        annotated_frame = results[0].plot()
        cv2.imshow('Yolo', annotated_frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()