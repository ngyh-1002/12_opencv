from ultralytics import YOLO
import cv2  

class TrafficMonitor:
    def __init__(self):
        self.model = YOLO('yolo11n.pt')
        self.traffic_classes = {
            0: 'person', 2: 'car', 3: 'motorcycle', 
            5: 'bus', 7: 'truck', 9: 'traffic_light'
        }
        self.stats = {'total_detections': 0, 'by_class': {}}
    
    def analyze_frame(self, frame):
        results = self.model(frame, 
                           classes=list(self.traffic_classes.keys()), 
                           conf=0.5, verbose=False)
        
        frame_stats = {'vehicles': 0, 'pedestrians': 0, 'signals': 0}
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = self.traffic_classes[class_id]
                
                if class_id in [2, 3, 5, 7]:  # vehicles
                    frame_stats['vehicles'] += 1
                elif class_id == 0:  # person
                    frame_stats['pedestrians'] += 1
                elif class_id == 9:  # traffic_light
                    frame_stats['signals'] += 1
                
                self.stats['by_class'][class_name] = \
                    self.stats['by_class'].get(class_name, 0) + 1
            
            self.stats['total_detections'] += len(results[0].boxes)
        
        return results[0].plot(), frame_stats
    
    def run_live_monitoring(self):
        cap = cv2.VideoCapture(0)
        print("🚗 교통 모니터링 시작! q로 종료")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame, frame_stats = self.analyze_frame(frame)
            
            # 정보 표시
            y = 30
            cv2.putText(annotated_frame, f"Vehicles: {frame_stats['vehicles']}", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
            cv2.putText(annotated_frame, f"Pedestrians: {frame_stats['pedestrians']}", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
            cv2.putText(annotated_frame, f"Total Detected: {self.stats['total_detections']}", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow('Traffic Monitoring System', annotated_frame)
            
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
monitor = TrafficMonitor()
monitor.run_live_monitoring()