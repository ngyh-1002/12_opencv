from ultralytics import YOLO
import cv2 

class dogmonitor:
    def __init__(self):
        self.model = YOLO('yolo11x.pt')
        self.traffic_classes = {
            16: 'dog'
        }
        self.stats = {'total_detections': 0, 'by_class': {}}
    
    def analyze_frame(self, frame):
        # model() 대신 model.track()을 사용하여 객체 추적 활성화
        # persist=True: 이전 프레임의 추적 정보를 유지
        # conf=0.3: 신뢰도 임계값을 낮춰 더 많은 객체를 추적하도록 함
        results = self.model.track(
            frame, 
            classes=list(self.traffic_classes.keys()), 
            conf=0.3, 
            persist=True, 
            verbose=False
        )
        
        frame_stats = {'dogs': 0}
        
        if results[0].boxes is not None:
            # 추적 ID를 포함한 바운딩 박스 정보 처리
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = self.traffic_classes[class_id]
                
                if class_id == 16:
                    frame_stats['dogs'] += 1
                
                self.stats['by_class'][class_name] = \
                    self.stats['by_class'].get(class_name, 0) + 1
            
            # 여기서 total_detections는 각 프레임에서 인식된 객체의 총합
            self.stats['total_detections'] += len(results[0].boxes)
        
        return results[0].plot(), frame_stats
    
    def run_video_monitoring(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"오류: 비디오 파일을 열 수 없습니다. 경로를 확인하세요: {video_path}")
            return
            
        print(f"🐶 비디오 파일 '{video_path}' 분석 시작! q로 종료")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame_full, frame_stats = self.analyze_frame(frame)
            
            height, width = annotated_frame_full.shape[:2]
            resized_annotated_frame = cv2.resize(annotated_frame_full, (width // 2, height // 2))

            y = 30
            cv2.putText(resized_annotated_frame, f"Dogs: {frame_stats['dogs']}", 
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
            cv2.putText(resized_annotated_frame, f"Total Detected: {self.stats['total_detections']}", 
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow('Dog Monitoring System', resized_annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.show_final_stats()
    
    def show_final_stats(self):
        print("\n📊 최종 통계:")
        print(f"총 검출 횟수: {self.stats['total_detections']}")
        print("클래스별 검출 현황:")
        for class_name, count in self.stats['by_class'].items():
            print(f"  {class_name}: {count}회")

# 시스템 실행
monitor = dogmonitor()
video_file_path = '../img/Idiot_dogs.mp4'
monitor.run_video_monitoring(video_file_path)