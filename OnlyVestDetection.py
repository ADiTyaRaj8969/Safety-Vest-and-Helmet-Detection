import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

# Load models
VEST_MODEL_PATH = r"D:\Internship\AI GURU Internship\Final\Q1\runs\detect\vest_helmet_final\weights\best.pt"
if not os.path.exists(VEST_MODEL_PATH):
    print(f"Error: Vest model not found at {os.path.abspath(VEST_MODEL_PATH)}")
    exit()

vest_model = YOLO(VEST_MODEL_PATH)
person_model = YOLO('yolov8n.pt')  # For person detection

# Configuration
DETECTION_CONFIDENCE = 0.6  # Increased confidence threshold
PERSON_PADDING = 15  # Reduced padding
DISPLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_DISPLAY_WIDTH = 800  # Max width for display

# Colors
COLOR_VEST = (0, 255, 0)      # Green
COLOR_NO_VEST = (0, 0, 255)    # Red
COLOR_INFO = (0, 255, 255)     # Yellow for info text
COLOR_FPS = (255, 255, 0)      # Cyan for FPS

class FPS_counter:
    def __init__(self):
        self.prev_time = 0
        self.curr_time = 0
        self.fps = 0
    
    def update(self):
        self.curr_time = time.time()
        if self.prev_time > 0:  # Skip first frame
            self.fps = 1 / (self.curr_time - self.prev_time)
        self.prev_time = self.curr_time
        return self.fps

def process_frame(frame, fps_counter):
    """Process each frame and return results"""
    fps_counter.update()
    
    # Detect persons
    person_results = person_model(frame, classes=[0], conf=DETECTION_CONFIDENCE, verbose=False)
    person_boxes = person_results[0].boxes.xyxy.cpu().numpy().astype(int)
    
    vest_status = []
    vest_count = 0
    
    for box in person_boxes:
        x1, y1, x2, y2 = box
        
        # Create a tighter crop for vest detection
        person_width = x2 - x1
        person_height = y2 - y1
        
        # Calculate a proportional padding
        pad_x = max(5, int(person_width * 0.05))
        pad_y = max(5, int(person_height * 0.05))
        
        pad_x1 = max(0, x1 - pad_x)
        pad_y1 = max(0, y1 - pad_y)
        pad_x2 = min(frame.shape[1], x2 + pad_x)
        pad_y2 = min(frame.shape[0], y2 + pad_y)
        
        person_crop = frame[pad_y1:pad_y2, pad_x1:pad_x2]
        
        # Skip very small crops (distant people)
        if person_crop.size == 0 or min(person_crop.shape[:2]) < 20:
            continue
            
        # Check for vest with increased confidence
        vest_results = vest_model(person_crop, conf=DETECTION_CONFIDENCE, verbose=False)
        has_vest = False
        
        # Check only for vest class (class 2)
        for box in vest_results[0].boxes:
            if int(box.cls) == 2:  # Vest class
                # Get bounding box coordinates in crop
                vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
                
                # Calculate vest size relative to person
                vest_width = vx2 - vx1
                vest_height = vy2 - vy1
                vest_area = vest_width * vest_height
                person_area = person_width * person_height
                
                # Only consider significant vest detections
                if vest_area > 0.05 * person_area:  # Vest must cover at least 5% of person area
                    has_vest = True
                    break
        
        if has_vest:
            vest_count += 1
        
        vest_status.append({
            'box': (x1, y1, x2, y2),
            'vest': has_vest
        })
    
    return vest_status, len(vest_status), vest_count, fps_counter.fps

def draw_results(frame, results, total_persons, vest_count, fps):
    """Draw all information on the frame"""
    # Draw person boxes and vest status
    for person in results:
        x1, y1, x2, y2 = person['box']
        color = COLOR_VEST if person['vest'] else COLOR_NO_VEST
        label = "Vest" if person['vest'] else "No Vest"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        cv2.rectangle(frame, (x1, y1-25), (x1+100, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1-5), DISPLAY_FONT, 0.6, (0, 0, 0), 2)
    
    # Draw information panel
    info_y = 30
    line_height = 30
    
    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, info_y), 
               DISPLAY_FONT, 0.7, COLOR_FPS, 2)
    info_y += line_height
    
    # Person count
    cv2.putText(frame, f"Total Persons: {total_persons}", (10, info_y), 
               DISPLAY_FONT, 0.7, COLOR_INFO, 2)
    info_y += line_height
    
    # Vest count
    cv2.putText(frame, f"With Vests: {vest_count}", (10, info_y), 
               DISPLAY_FONT, 0.7, COLOR_VEST, 2)
    info_y += line_height
    
    # No vest count
    cv2.putText(frame, f"Without Vests: {total_persons - vest_count}", (10, info_y), 
               DISPLAY_FONT, 0.7, COLOR_NO_VEST, 2)
    
    return frame

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  # Balanced resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    
    fps_counter = FPS_counter()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        results, total_persons, vest_count, fps = process_frame(frame, fps_counter)
        
        # Draw results
        frame = draw_results(frame, results, total_persons, vest_count, fps)
        
        # Resize for display if too wide
        if frame.shape[1] > MAX_DISPLAY_WIDTH:
            scale = MAX_DISPLAY_WIDTH / frame.shape[1]
            display_frame = cv2.resize(frame, (MAX_DISPLAY_WIDTH, int(frame.shape[0] * scale)))
        else:
            display_frame = frame
        
        # Display
        cv2.imshow("Vest Detection System", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()