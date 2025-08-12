from ultralytics import YOLO
import cv2  

# YOLO 모델 로드
model = YOLO('yolo11n.pt')

# 신뢰도별 검출 결과 비교
confidence_levels = [0.25, 0.5, 0.75]
test_image = 'https://ultralytics.com/images/bus.jpg'

print("🧪 신뢰도별 검출 실험:")
for conf in confidence_levels:
    results = model(test_image, conf=conf, verbose=False)
    num_objects = len(results[0].boxes) if results[0].boxes else 0
    print(f"신뢰도 {conf}: {num_objects}개 객체 검출")

# 실시간 신뢰도 조정 도구
confidence = 0.5
cap = cv2.VideoCapture(0)

print("키보드 조작: +/- 로 신뢰도 조정, q로 종료")
while True:
    ret, frame = cap.read()
    if ret:
        results = model(frame, conf=confidence, verbose=False)
        annotated = results[0].plot()
        
        # 현재 설정 표시
        info_text = f"Confidence: {confidence:.2f}"
        cv2.putText(annotated, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Confidence Tuner', annotated)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        confidence = min(0.95, confidence + 0.1)
    elif key == ord('-'):
        confidence = max(0.1, confidence - 0.1)