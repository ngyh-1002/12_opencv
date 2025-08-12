from ultralytics import YOLO
import cv2 

class dogmonitor:
    def __init__(self):
        self.model = YOLO('yolo11n.pt')
        # 오직 'dog' (클래스 ID: 16)만 인식하도록 수정
        self.traffic_classes = {
            16: 'dog'
        }
        self.stats = {'total_detections': 0, 'by_class': {}}
    
    def analyze_frame(self, frame):
        # 감지할 클래스를 'dog'로 한정
        results = self.model(frame, 
                             classes=list(self.traffic_classes.keys()), 
                             conf=0.1, verbose=False)
        
        # 'dogs'만 포함하는 통계 딕셔너리
        frame_stats = {'dogs': 0}
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = self.traffic_classes[class_id]
                
                # 감지된 객체가 'dog'일 경우에만 통계 업데이트
                if class_id == 16:
                    frame_stats['dogs'] += 1
                
                self.stats['by_class'][class_name] = \
                    self.stats['by_class'].get(class_name, 0) + 1
            
            self.stats['total_detections'] += len(results[0].boxes)
        
        return results[0].plot(), frame_stats
    
    def run_video_monitoring(self, video_path):
        """웹캠 대신 비디오 파일을 분석하도록 변경된 메서드"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"오류: 비디오 파일을 열 수 없습니다. 경로를 확인하세요: {video_path}")
            return
            
        print(f"🐶 비디오 파일 '{video_path}' 분석 시작! q로 종료")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            height, width = frame.shape[:2]
            resized_frame = cv2.resize(frame, (width // 2, height // 2))
            annotated_frame, frame_stats = self.analyze_frame(resized_frame)
            
            # 정보 표시
            y = 30
            cv2.putText(annotated_frame, f"Dogs: {frame_stats['dogs']}", 
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
            cv2.putText(annotated_frame, f"Total Detected: {self.stats['total_detections']}", 
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow('Dog Monitoring System', annotated_frame)
            
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
# 여기에 분석하고 싶은 비디오 파일의 경로를 넣어주세요.
# 예를 들어, 'dog_video.mp4' 파일이 현재 스크립트와 같은 폴더에 있다면
video_file_path = '../img/Idiot_dogs.mp4' 
monitor.run_video_monitoring(video_file_path)