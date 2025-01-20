from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-obb.pt")
# Predict with the model

results = model('boats.jpg') # predict on an image
print(results)