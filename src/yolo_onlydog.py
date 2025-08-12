from ultralytics import YOLO
import cv2 

class dogmonitor:
    def __init__(self):
        self.model = YOLO('yolo11s.pt')
        self.traffic_classes = {
            16: 'dog'
        }
        self.stats = {'total_detections': 0, 'by_class': {}}
    
    def analyze_frame(self, frame):
        # 1. 원본 프레임에서 객체 인식 수행
        results = self.model(frame, 
                             classes=list(self.traffic_classes.keys()), 
                             conf=0.3, verbose=False)
        
        frame_stats = {'dogs': 0}
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = self.traffic_classes[class_id]
                
                if class_id == 16:
                    frame_stats['dogs'] += 1
                
                self.stats['by_class'][class_name] = \
                    self.stats['by_class'].get(class_name, 0) + 1
            
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
            
            # 2. 원본 프레임에서 인식 후, 결과를 시각화
            # 'results[0].plot()' 함수가 이미 원본 프레임에 바운딩 박스를 그려서 반환하므로,
            # 별도의 좌표 변환 과정이 필요 없습니다.
            annotated_frame_full, frame_stats = self.analyze_frame(frame)
            
            # 3. 시각화된 프레임의 크기를 절반으로 줄여서 출력
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