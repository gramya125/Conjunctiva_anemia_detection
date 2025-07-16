# Conjunctiva Anemia Detection

This project provides a deep learning pipeline for detecting anemia using eye images by focusing on the conjunctiva region. It uses a two-stage approach:

1. **Conjunctiva Detection**: A YOLOv8 model detects and crops the conjunctiva from both left and right eye images.
2. **Anemia Classification**: A CNN-based classifier (e.g., MobileNetV2 or ResNet) predicts whether the person is anemic based on the cropped region.

## Components

- `anemia_detection.py`: Runs detection and classification on a single image.
- `app.py`: Flask API that accepts left and right eye images and returns the predicted anemia label and cropped image in base64 format.
- `models/`: Contains trained model files (`.pt`, `.pth`), tracked via Git LFS.
- `test_images`: Stores cropped conjunctiva images.
- `app.py`: Helper functions for YOLOv8 detection and classification.

## Setup

### Clone the repository

```bash
git clone https://github.com/gramya125/Conjunctiva_anemia_detection.git
cd Conjunctiva_anemia_detection
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Install Git LFS and pull large files

```bash
git lfs install
git lfs pull
```

## Usage

### Run on a Local Image

```bash
python anemia_pipeline.py --image path/to/image.jpg
```

### Run the Flask API

```bash
python app.py
```

Send a POST request to `/predict` with form-data fields:

- `left_eye`: image file  
- `right_eye`: image file

The response includes:

- Predicted label: `Anemia` or `Non-Anemia`
- Base64-encoded cropped conjunctiva image

## Notes

- Model files larger than 100 MB are tracked using Git LFS.
- This project is for research and development purposes only.

## Contact

Author: gramya125  
Email: g25ramya@gmail.com  
GitHub: [https://github.com/gramya125](https://github.com/gramya125)
