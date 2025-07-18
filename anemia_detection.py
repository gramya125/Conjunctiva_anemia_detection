import os
import cv2
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import matplotlib.pyplot as plt

# === Setup paths ===
LEFT_IMAGE_PATH = "test_images/jamileft.jpg"
RIGHT_IMAGE_PATH = "test_images/jamiright.jpg"
YOLO_MODEL_PATH = "models/yolov8_model.pt"
RESNET_MODEL_PATH = "models/resnet101_anemia_model.pth"

# === Load YOLOv8 model ===
yolo_model = YOLO(YOLO_MODEL_PATH)

# === Load ResNet Model ===
checkpoint = torch.load(RESNET_MODEL_PATH, map_location='cpu')
from torchvision.models import resnet101
model = resnet101(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
class_names = ["Non Anemia", "Anemia"]

# === Helper: Crop conjunctiva using YOLO ===
def crop_conjunctiva(image_path):
    image = cv2.imread(image_path)
    results = yolo_model.predict(image_path, conf=0.25)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        print(f"❌ No conjunctiva detected in: {image_path}")
        return None

    # Only take first detection
    x1, y1, x2, y2 = map(int, boxes[0])
    cropped = image[y1:y2, x1:x2]
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    # Show cropped image
    plt.imshow(cropped_rgb)
    plt.axis("off")
    plt.title(f"Cropped: {os.path.basename(image_path)}")
    plt.show()

    return cropped_rgb

# === Helper: Classify cropped image ===
def classify_anemia(cropped_img):
    pil_img = Image.fromarray(cropped_img)
    input_tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        return class_names[pred.item()]

# === Main Execution ===
def process_both_eyes():
    predictions = []

    # Left Eye
    left_crop = crop_conjunctiva(LEFT_IMAGE_PATH)
    if left_crop is not None:
        left_result = classify_anemia(left_crop)
        predictions.append(left_result)
        print(f"👁️ Left Eye: {left_result}")

    # Right Eye
    right_crop = crop_conjunctiva(RIGHT_IMAGE_PATH)
    if right_crop is not None:
        right_result = classify_anemia(right_crop)
        predictions.append(right_result)
        print(f"👁️ Right Eye: {right_result}")

    # Final decision
    if "Anemia" in predictions:
        print("\n🩸 Final Diagnosis: Anemia")
    elif "Non Anemia" in predictions and predictions:
        print("\n🩸 Final Diagnosis: Non Anemia")
    else:
        print("\n⚠️ No conjunctiva detected in both images.")

if __name__ == "__main__":
    process_both_eyes()
