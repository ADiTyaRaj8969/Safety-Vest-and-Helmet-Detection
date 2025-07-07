from ultralytics import YOLO
import os
import shutil

# Config
DATA_YAML = "Q1/data.yaml"
MODEL_NAME = "vest_helmet_final"
RESUME_TRAINING = False  # Set True if you want to resume previous training

def train_model():
    # Load model
    model = YOLO("yolov8n.pt")  # Base model
    
    # Train
    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=8,
        device="cpu",
        project="Q1/runs/detect",
        name=MODEL_NAME,
        exist_ok=True,
        augment=True  # Enable basic augmentations
    )
    
    # Verify output
    model_path = f"Q1/runs/detect/{MODEL_NAME}/weights/best.pt"
    if os.path.exists(model_path):
        print(f"\nTraining successful! Model saved to:\n{os.path.abspath(model_path)}")
        return model_path
    else:
        raise FileNotFoundError("Model weights not generated")

if __name__ == "__main__":
    train_model()
