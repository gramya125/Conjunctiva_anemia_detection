import os
import cv2
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ========== PATHS ==========
YOLO_MODEL_PATH = r"C:\Users\g125r\OneDrive\Documents\ANEMIA INTEGRATION CONJUCTIVA\yolov8_conjunctiva_best.pt"
RESNET_MODEL_PATH = r"C:\Users\g125r\OneDrive\Documents\ANEMIA INTEGRATION CONJUCTIVA\resnet101_anemia_model1.pth"
LEFT_EYE_IMAGE = r"C:\Users\g125r\OneDrive\Documents\ANEMIA INTEGRATION CONJUCTIVA\images\left_eye.jpg"
RIGHT_EYE_IMAGE = r"C:\Users\g125r\OneDrive\Documents\ANEMIA INTEGRATION CONJUCTIVA\images\right_eye.jpg"

# ========== SETUP ==========
class_names = ['Non Anemia', 'Anemia']

# Load YOLOv8 model
yolo_model = YOLO(YOLO_MODEL_PATH)

# Load ResNet model
checkpoint = torch.load(RESNET_MODEL_PATH, map_location='cpu')
if 'model_state_dict' in checkpoint:
    from torchvision.models import resnet101
    model = resnet101(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model = checkpoint
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ========== FUNCTIONS ==========
def crop_conjunctiva(image_path):
    image = cv2.imread(image_path)
    results = yolo_model.predict(image_path, conf=0.25)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        print(f"❌ No conjunctiva detected in: {image_path}")
        return None

    x1, y1, x2, y2 = map(int, boxes[0])  # Only the first detection
    cropped = image[y1:y2, x1:x2]
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    # Display the cropped image
    plt.figure(figsize=(3, 3))
    plt.imshow(cropped_rgb)
    plt.title(f"Cropped Conjunctiva")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return cropped_rgb

def classify_anemia(cropped_rgb):
    pil_img = Image.fromarray(cropped_rgb)
    input_tensor = transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        return class_names[pred.item()]

def predict_anemia_from_eyes(left_eye_path, right_eye_path):
    left_crop = crop_conjunctiva(left_eye_path)
    right_crop = crop_conjunctiva(right_eye_path)

    if left_crop is None and right_crop is None:
        return "No conjunctiva detected in either image."

    predictions = []
    if left_crop is not None:
        left_pred = classify_anemia(left_crop)
        predictions.append(left_pred)
        print(f"👁️ Left Eye Prediction: {left_pred}")

    if right_crop is not None:
        right_pred = classify_anemia(right_crop)
        predictions.append(right_pred)
        print(f"👁️ Right Eye Prediction: {right_pred}")

    # Final Decision
    if "Anemia" in predictions:
        return "🩸 FINAL RESULT: Anemia"
    else:
        return "🩸 FINAL RESULT: Non Anemia"

# ========== RUN ==========
if __name__ == "__main__":
    result = predict_anemia_from_eyes(LEFT_EYE_IMAGE, RIGHT_EYE_IMAGE)
    print("\n🔍", result)
