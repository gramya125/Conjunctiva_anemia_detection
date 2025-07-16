# Conjunctiva Anemia Detection

> ⚠️ **Note**: This system is under active development. It is not intended for clinical use.

An AI-powered system that predicts anemia using images of the conjunctiva (inner eyelid). The pipeline uses a YOLOv8 model to detect and crop the conjunctiva region from left and right eye images, followed by a CNN-based classifier to determine anemia status. The system supports both command-line testing and REST API integration.

---

## 🔍 Key Features

- **Conjunctiva Detection** – YOLOv8 model auto-detects and crops inner eyelid (conjunctiva)
- **Anemia Classification** – CNN (MobileNetV2 or ResNet) predicts anemia from cropped images
- **Dual Image Support** – Accepts both left and right eye inputs
- **REST API** – Flask-based API for web and mobile integration
- **Cropped Image Output** – Returns base64-encoded cropped conjunctiva images

---

## 📌 Architecture

Eye Image (Left & Right)  
&nbsp;&nbsp;&nbsp;&nbsp;↓  
**YOLOv8** → Cropped Conjunctiva  
&nbsp;&nbsp;&nbsp;&nbsp;↓  
**CNN Classifier** → Prediction (`Anemia` / `Non-Anemia`)

---

## 🗂️ Project Structure

```
conjunctiva_anemia_detection/
├── anemia_pipeline.py         # CLI script for local testing
├── app.py                     # Flask API server
├── models/                    # YOLOv8 & classifier model files
│   ├── yolo_conjunctiva.pt
│   └── anemia_classifier.pth
├── static/                    # Stores cropped conjunctiva images
├── utils/                     # YOLO & classification utilities
├── test_images/               # Sample eye images for testing
├── requirements.txt
└── README.md
```

---

## ⚙️ Quick Start

### 🔧 Prerequisites

- Python 3.8+
- pip (Python package manager)

### 📥 Installation

```bash
git clone https://github.com/gramya125/Conjunctiva_anemia_detection.git
cd Conjunctiva_anemia_detection

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Git LFS and pull model files
git lfs install
git lfs pull
```

---

## 🚀 Usage

### Method 1: Command Line Interface (CLI)

```bash
python anemia_pipeline.py --image path/to/eye_image.jpg
```

- Detects and crops conjunctiva
- Runs anemia classification
- Saves cropped image and displays prediction

### Method 2: REST API (Flask)

```bash
python app.py
```

Send a `POST` request to `http://localhost:5000/predict` with form-data fields:

- `left_eye`: image file  
- `right_eye`: image file

Response includes:

- Predicted Label: `Anemia` or `Non-Anemia`
- Base64-encoded cropped image

---

## 🛠 Requirements

```text
ultralytics==8.0.192
flask==2.3.2
torch>=2.0.0
torchvision>=0.15.0
opencv-python
numpy
Pillow
matplotlib
scikit-learn
flask-cors
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 📬 Contact

Author: gramya125  
Email: g25ramya@gmail.com  
GitHub: [https://github.com/gramya125](https://github.com/gramya125)

---

## ⚠️ Medical Disclaimer

This project is intended for research and educational purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals.
