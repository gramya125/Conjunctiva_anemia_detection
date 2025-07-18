import os
import cv2
import torch
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from torchvision import transforms
from ultralytics import YOLO
from torchvision.models import resnet101

# === Flask setup ===
app = Flask(__name__)
UPLOAD_FOLDER = "test_images"
CROPPED_FOLDER = "cropped"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)

# === Load YOLOv8 Model ===
yolo_model = YOLO("models/yolov8_model.pt")

# === Load ResNet Model ===
checkpoint = torch.load("models/resnet101_anemia_model1.pth", map_location='cpu')
model = resnet101(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
class_names = ["Non Anemia", "Anemia"]

# === Helper: crop conjunctiva, return RGB image and base64 ===
def crop_and_classify(img_np, eye_side):
    results = yolo_model.predict(img_np, conf=0.25)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        return {
            "eye": eye_side,
            "prediction": "No conjunctiva detected",
            "confidence": 0.0,
            "cropped_base64": None
        }

    x1, y1, x2, y2 = map(int, boxes[0])
    cropped = img_np[y1:y2, x1:x2]
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    # Save .png
    crop_path = os.path.join(CROPPED_FOLDER, f"{eye_side}_crop.png")
    cv2.imwrite(crop_path, cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR))

    # Save .txt (base64)
    _, buffer = cv2.imencode('.png', cropped_rgb)
    base64_crop = base64.b64encode(buffer).decode('utf-8')
    with open(os.path.join(CROPPED_FOLDER, f"{eye_side}_crop.txt"), "w") as f:
        f.write(base64_crop)

    # Predict
    pil_img = Image.fromarray(cropped_rgb)
    input_tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        pred_idx = probs.argmax().item()
        confidence = round(probs[pred_idx].item(), 4)
        prediction = class_names[pred_idx]

    return {
        "eye": eye_side,
        "prediction": prediction,
        "confidence": confidence,
        "cropped_base64": base64_crop
    }

# === Flask Endpoint ===
@app.route("/predict", methods=["POST"])
def predict_dual_eye():
    if 'left_eye' not in request.files or 'right_eye' not in request.files:
        return jsonify({"error": "Both 'left_eye' and 'right_eye' files are required"}), 400

    response = {}

    # Process Left Eye
    left_file = request.files['left_eye']
    left_npimg = np.frombuffer(left_file.read(), np.uint8)
    left_img = cv2.imdecode(left_npimg, cv2.IMREAD_COLOR)
    left_result = crop_and_classify(left_img, "left_eye")

    # Process Right Eye
    right_file = request.files['right_eye']
    right_npimg = np.frombuffer(right_file.read(), np.uint8)
    right_img = cv2.imdecode(right_npimg, cv2.IMREAD_COLOR)
    right_result = crop_and_classify(right_img, "right_eye")

    # Aggregate final result
    final = "Anemia" if "Anemia" in [left_result["prediction"], right_result["prediction"]] else "Non Anemia"

    response = {
        "left_eye": left_result,
        "right_eye": right_result,
        "aggregation_method": "Anemia if any eye shows Anemia",
        "final_prediction": final
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, port=2000)
