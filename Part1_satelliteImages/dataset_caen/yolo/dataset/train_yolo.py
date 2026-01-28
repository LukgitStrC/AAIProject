from ultralytics import YOLO


if __name__ == '__main__':


    model = YOLO("yolo11n-seg.pt")

    results = model.train(data="data.yaml", epochs=300, imgsz=512, batch=16, device=0)