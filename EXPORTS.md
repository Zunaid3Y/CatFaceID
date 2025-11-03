### CatFaceID Export Cheatsheet

Run these from the project root after training:

```bash
python3 -m ultralytics export model=runs/detect/cat_head/weights/best.pt format=onnx
python3 -m ultralytics export model=runs/detect/cat_head/weights/best.pt format=coreml
python3 -m ultralytics export model=runs/detect/cat_head/weights/best.pt format=tflite
```

All variants expect 640×640 RGB images by default (match your training `imgsz`). On-device you’ll need to apply the same preprocessing and run non-max suppression (NMS) on the exported outputs to filter overlapping detections.
