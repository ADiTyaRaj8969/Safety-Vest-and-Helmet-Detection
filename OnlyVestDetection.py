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
DISPLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_DISPLAY_WIDTH = 800  # Max width for display

# Colors
COLOR_VEST = (0, 255, 0)      # Green
COLOR_NO_VEST = (0, 0, 255)   # Red
COLOR_INFO = (0, 255, 255)    # Yellow for info text
COLOR_FPS = (255, 255, 0)     # Cyan for FPS
COLOR_CONF = (255, 255, 255)  # White for confidence text

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
        
        # Create a crop for vest detection
        person_crop = frame[y1:y2, x1:x2]
        
        # Skip very small crops (distant people)
        if person_crop.size == 0 or min(person_crop.shape[:2]) < 20:
            continue
            
        # Check for vest
        vest_results = vest_model(person_crop, conf=DETECTION_CONFIDENCE, verbose=False)
        max_vest_conf = 0.0
        
        # Check only for vest class (class 2)
        for box in vest_results[0].boxes:
            if int(box.cls) == 2:  # Vest class
                conf = float(box.conf.item())
                if conf > max_vest_conf:  
                    max_vest_conf = conf
        
        has_vest = max_vest_conf > 0
        if has_vest:
            vest_count += 1
        
        vest_status.append({
            'box': (x1, y1, x2, y2),
            'vest': has_vest,
            'vest_conf': max_vest_conf
        })
    
    return vest_status, len(vest_status), vest_count, fps_counter.fps

def draw_results(frame, results, total_persons, vest_count, fps):
    """Draw all information on the frame with single box per person"""
    # Draw person boxes and vest status
    for person in results:
        x1, y1, x2, y2 = person['box']
        color = COLOR_VEST if person['vest'] else COLOR_NO_VEST
        
        # Draw single bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare status text
        if person['vest']:
            status_text = f"Vest: {person['vest_conf']:.2f}"
        else:
            status_text = "No Vest"
        
        # Calculate text size for background
        text_size = cv2.getTextSize(status_text, DISPLAY_FONT, 0.6, 2)[0]
        text_width = text_size[0] + 10
        
        # Draw text background
        cv2.rectangle(frame, (x1, y1-30), (x1 + text_width, y1), color, -1)
        
        # Draw status text
        cv2.putText(frame, status_text, (x1+5, y1-10), 
                   DISPLAY_FONT, 0.6, (0, 0, 0), 2)
    
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
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
