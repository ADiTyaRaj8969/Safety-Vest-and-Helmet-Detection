# 🦺 Safety Equipment Detection using YOLOv8

This project performs **real-time detection** of safety helmets and vests on individuals using **YOLOv8**. It helps monitor compliance with safety protocols in environments like construction sites and factories.

---

## 🚀 Features

- 🔍 Detects **helmets** and **vests**
- 👷‍♂️ Classifies people as:
  - ✅ SAFE (Helmet + Vest)
  - ⚠️ PARTIAL (Only Helmet or Only Vest)
  - ❌ UNSAFE (No safety gear)
- 🎨 Color-coded bounding boxes:
  - Green = Safe
  - Yellow = Vest only
  - Cyan = Helmet only
  - Red = No gear
- 📹 Real-time webcam or video input
- 🧠 Trainable on custom datasets

---

## 📁 Project Structure


├── safety_detection.py # Inference script with bounding boxes<br>
├── train.py # Model training using YOLOv8<br>
├── yolov8n.pt # Pretrained YOLOv8 model<br>
└── Q1/<br>
├── data.yaml # Dataset configuration<br>
└── runs/<br>
└── detect/<br>
└── vest_helmet_final/<br>
└── weights/<br>
└── best.pt # Trained model output<br>


---

## 🔗 Connect with Me

- 🔗 GitHub: [@ADiTyaRaj8969](https://github.com/ADiTyaRaj8969)  
- ✉️ Email: adivid198986@gmail.com  
- 💼 LinkedIn: [Aditya Raj](https://www.linkedin.com/in/aditya-raj-710a5a291/)

---
## 📦 Installation

Install required packages:

```bash
pip install ultralytics opencv-python numpy
#Training the Model
Make sure your dataset is annotated and referenced in Q1/data.yaml.

To start training:
bash

python train.py
The best model weights will be saved to:
swift

Q1/runs/detect/vest_helmet_final/weights/best.pt
 Running Inference
Open safety_detection.py and ensure this path is set:

python

MODEL_PATH = "Q1/runs/detect/vest_helmet_final/weights/best.pt"
Run the detection:

bash

python safety_detection.py
Opens webcam by default (cv2.VideoCapture(0))

Press q to quit

🎨 Bounding Box Colors
Status	Description	Box Color
✅ SAFE	Helmet + Vest	Green
⚠️ PARTIAL	Helmet only / Vest only	Cyan / Yellow
❌ UNSAFE	No helmet, no vest	Red

🏷️ Class Labels (YOLO Format)
Make sure your dataset uses the following labels:

0: No Vest

1: Helmet

2: Vest

These labels are used in both training and detection.

🧰 Technologies Used
YOLOv8 (Ultralytics)

OpenCV for video processing

Python 3

📸 Output Preview
Add sample screenshots or video demos here for better presentation.

🙌 Contributing
Pull requests are welcome! For major changes, open an issue first to discuss what you would like to change.

📬 Contact
For suggestions or queries, feel free to raise an issue in this repository.

