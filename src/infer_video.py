import argparse

import cv2
import torch

from .model import TinyYOLOv1
from .utils import decode_predictions, nms_normalized

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="runs/yolov1_tiny_pennfudan.pt")
    p.add_argument("--source", type=str, required=True, help="video path or webcam index (0)")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.5)
    return p.parse_args()

def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = ckpt["cfg"]

    S = int(cfg["S"]); C = int(cfg["C"]); img_size = int(cfg["img_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyYOLOv1(S=S, C=C).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h0, w0 = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rs = cv2.resize(rgb, (img_size, img_size))

        x = torch.from_numpy(rs).float() / 255.0
        x = x.permute(2,0,1).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x)[0]

        boxes, scores = decode_predictions(pred, S=S, conf_thresh=args.conf)
        keep = nms_normalized(boxes, scores, iou_thresh=args.iou)
        boxes = boxes[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy()

        for (x1,y1,x2,y2), sc in zip(boxes, scores):
            X1 = int(x1 * w0); Y1 = int(y1 * h0)
            X2 = int(x2 * w0); Y2 = int(y2 * h0)
            cv2.rectangle(frame, (X1,Y1), (X2,Y2), (0,255,0), 2)
            cv2.putText(frame, f"person {sc:.2f}", (X1, max(0, Y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("detections", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
