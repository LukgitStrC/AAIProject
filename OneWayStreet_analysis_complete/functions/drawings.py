import cv2
from .p5_streetSigns import StreetLabel
import math


def get_color(label):
  match label:
    case StreetLabel.ONE_WAY_ENTRY.value:
      return (255, 0, 0) # red
    case StreetLabel.ONE_WAY_EXIT.value:
      return (0, 0, 255) # red
    case StreetLabel.BOTH.value:
      return (238, 130, 238) # violet
    case StreetLabel.NONE.value:
      return (128,128,128)# grey
    case _:
      return (128,128,128)# grey

def create_marked_svp(input_path, output_path, roads, alpha=0.5):
  img = cv2.imread(input_path)
  if img is None:
      print("Error: Could not load image.")
      return
  h, w = img.shape[:2]
  overlay = img.copy()

  start_point = (int(w / 2), int(h * 0.9))

  for road in roads:
      end_x = int((road['yaw'] % 360) * (w / 360))
      end_y = int(h * 0.5)
      color = get_color(road['label'])
      cv2.arrowedLine(overlay,
              start_point,
              (end_x, end_y), 
              color,
              thickness=max(2, int(w/200)),
              tipLength=0.1)
  cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
  cv2.imwrite(output_path, img)