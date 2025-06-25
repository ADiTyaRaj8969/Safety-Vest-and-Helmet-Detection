import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import defaultdict

# Load the trained model
MODEL_PATH = r"D:\Internship\AI GURU Internship\Final\Q1\runs\detect\vest_helmet_final\weights\best.pt"

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {os.path.abspath(MODEL_PATH)}")
    exit()
model = YOLO(MODEL_PATH)

# Define color codes for different safety states
COLORS = {
    "safe": (0, 255, 0),        # Green - Both vest and helmet
    "vest_only": (0, 255, 255),  # Yellow - Only vest
    "helmet_only": (255, 255, 0),  # Cyan - Only helmet
    "unsafe": (0, 0, 255)       # Red - No safety equipment
}

# Define class names based on your training
CLASS_NAMES = {
    0: "No Vest",
    1: "Helmet",
    2: "Vest"
}

def get_person_status(detections):
    """Group detections by person and determine safety status"""
    # Group detections by person using center points
    person_status = defaultdict(lambda: {"vest": False, "helmet": False, "boxes": []})
    
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        class_id = int(box.cls)
        
        # Find closest person within a reasonable distance
        person_id = None
        min_distance = float('inf')
        for pid, pdata in person_status.items():
            pcenter = pdata["center"]
            distance = np.sqrt((center[0]-pcenter[0])**2 + (center[1]-pcenter[1])**2)
            if distance < min_distance and distance < 100:  # 100 pixels max
                min_distance = distance
                person_id = pid
        
        # Create new person if not found
        if person_id is None:
            person_id = center
            person_status[person_id] = {"vest": False, "helmet": False, "center": center, "boxes": []}
        
        # Update safety status
        if class_id == 2:  # Vest
            person_status[person_id]["vest"] = True
        elif class_id == 1:  # Helmet
            person_status[person_id]["helmet"] = True
        
        # Update center position
        person_status[person_id]["center"] = center
        person_status[person_id]["boxes"].append((x1, y1, x2, y2))
    
    return person_status

def draw_safety_status(frame, person_status):
    """Draw bounding boxes and safety status on the frame"""
    for pid, status in person_status.items():
        # Find the largest box for this person
        if not status["boxes"]:
            continue
            
        largest_box = max(status["boxes"], key=lambda box: (box[2]-box[0])*(box[3]-box[1]))
        x1, y1, x2, y2 = largest_box
        
        # Determine safety status and color
        if status["vest"] and status["helmet"]:
            color = COLORS["safe"]
            label = "SAFE: Vest+Helmet"
        elif status["vest"]:
            color = COLORS["vest_only"]
            label = "PARTIAL: Vest Only"
        elif status["helmet"]:
            color = COLORS["helmet_only"]
            label = "PARTIAL: Helmet Only"
        else:
            color = COLORS["unsafe"]
            label = "UNSAFE: No Protection"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + len(label) * 10, y1), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or path to video file
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Run YOLOv8 inference
        results = model.predict(frame, conf=0.5, device="cpu")
        
        # Process detections
        detections = []
        for result in results:
            detections.extend(result.boxes)
        
        # Group detections by person and determine safety status
        person_status = get_person_status(detections)
        
        # Draw safety status on frame
        frame = draw_safety_status(frame, person_status)
        
        # Display the annotated frame
        cv2.imshow("Safety Vest & Helmet Detection", frame)
        
        # Break the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()