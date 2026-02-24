from ultralytics import YOLO
import skimage.io as io


# model used for sign detection
detect_face_plate_sign_model = YOLO("https://huggingface.co/Panoramax/detect_face_plate_sign/resolve/main/yolov8s_panoramax.pt")
detect_face_plate_sign_model.to('mps') # optional use the device that you have

# model used for classification
classification_model = YOLO("Panoramax/classify_fr_road_signs")
classification_model.to('mps')  # optional use the device that you have

# other devices are cpu (default) and cuda

def detect_and_classify(image_path):
  img = io.imread(image_path)
  det_results = detect_face_plate_sign_model(img)[0]
  final_results = []
  for box in det_results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = box.conf[0].item()
    crop = img[y1:y2, x1:x2]
    if crop.size > 0:
      cls_results = classification_model(crop)[0]
      top_class_idx = cls_results.probs.top1
      class_name = cls_results.names[top_class_idx]
      cls_conf = cls_results.probs.top1conf.item()
      final_results.append({
          "box": [x1, y1, x2, y2],
          "det_conf": conf,
          "label": class_name,
          "cls_conf": cls_conf
      })
  return final_results