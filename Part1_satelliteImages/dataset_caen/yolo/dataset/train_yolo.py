from ultralytics import YOLO
from ultralytics.data.split import autosplit



if __name__ == '__main__':


    model = YOLO("yolo11n-seg.pt")

    img_fp = "images"

    results = model.train(data="data.yaml", epochs=100, imgsz=512, batch=16, device=0)