from ultralytics import YOLO
import cv2
from enum import Enum

# model used for sign detection
detect_face_plate_sign_model = YOLO("https://huggingface.co/Panoramax/detect_face_plate_sign/resolve/main/yolov8s_panoramax.pt")
detect_face_plate_sign_model.to('cuda') # optional use the device that you have

# model used for classification
classification_model = YOLO("https://huggingface.co/Panoramax/classify_fr_road_signs/resolve/main/best.pt")
classification_model.to('cuda')  # optional use the device that you have

# other devices are cpu (default) and cuda

class StreetLabel(Enum):
  """
  Labels to classify each crop (and therefore each street entrance)
  """
  ONE_WAY_EXIT = 'ONE_WAY_EXIT'
  ONE_WAY_ENTRY = 'ONE_WAY_ENTRY'
  BOTH = 'OPAQUE_STREET'
  NONE = 'TWO_WAY_STREET'



def detect_and_classify(image_path):
  img = cv2.imread(image_path)
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
      print(f'{class_name} Sign detected')
      final_results.append({
          "box": [x1, y1, x2, y2],
          "det_conf": conf,
          "label": class_name,
          "cls_conf": cls_conf
      })
  return final_results

def classify_street(image_path):
  model_results = detect_and_classify(image_path)
  noentry_sign_visible = any([res['label'] == 'B1' for res in model_results])
  oneway_sign_visible = any([res['label'] == 'C1a' for res in model_results])
  if oneway_sign_visible:
    if noentry_sign_visible:
      return StreetLabel.BOTH
    else:
      return StreetLabel.ONE_WAY_ENTRY
  else:
    if noentry_sign_visible:
      return StreetLabel.ONE_WAY_EXIT
    else:
      return StreetLabel.NONE