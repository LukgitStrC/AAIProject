from ultralytics import YOLO
from ultralytics.data.split import autosplit



if __name__ == '__main__':


    model = YOLO("./yolo11n-seg.pt")

    img_fp = "dataset/images"
    autosplit(path=img_fp, weights=(0.8,0.15,0.05), annotated_only=True)

    results = model.train(data="data.yaml", epochs=100, imgsz=512, device=0)