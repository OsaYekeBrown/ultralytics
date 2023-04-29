# Ultralytics YOLO 🚀, GPL-3.0 license

import cv2
import hydra
import torch
import math
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        # save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        if self.return_outputs:
            self.output["det"] = det.cpu().numpy()

        # write
        pts = deque(maxlen=args["buffer"]) # Empty list
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
            c = int(cls)  # integer class
                label = None 
                # c1 = (x1, y1) or top-leftmost coordinate while; c2 = (x2, y2) or bottom-rightmost coordinate
                # c1 = (x1, y1) or top-leftmost coordinate while; c2 = (x2, y2) or bottom-rightmost coordinate
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]) , int(xyxy[3])) # extracting coordinates x1, y1, x2, y2
                cX = ((int(c1[0]) + int(c2[0])) / 2.0) # average of x1 and x2
                cY = ((int(c1[1]) + int(c2[1])) / 2.0) # average of y1 and y2
                center = (int(cX), int(cY))
                #pnt_bck = (int(cX-(cX/55)), int(cY-(cY/55)))
                #pnt_frwrd = (int(cX+(cX/55)), int(cY+(cY/55)))
                
                # only proceed if the radius meets a minimum size and the amount of points is beyond 2
                if len(pts) >= 2:
                    radius = math.dist(pts[0],pts[1])
                    if radius > 5:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(im0, center, int(radius), (0, 255, 255), 2)
                    cv2.circle(im0, center, 5, (0, 0, 255), -1)
                else:
                    cv2.circle(im0, center, 5, (0, 0, 255), -1)

                #cv2.circle(im0, pnt, 31, (255,18,120), -1) # under circle
                #cv2.circle(im0, pnt, 20, (252, 15, 190), -1) # visualization if centroid in video! :)
                #cv2.circle(im0, pnt_bck, 25, (255,18,125), -1) # line points
                #cv2.circle(im0, pnt_frwrd, 25, (255,18,125), -1) # line points
                
                pts.appendleft(center)
                print(pts)
                # loop over the set of tracked points
                for i in range(1, len(pts)): 
                    # if either of the tracked points are None, ignore
                    # them
                    if pts[i - 1] is None or pts[i] is None:
                        continue

                    # otherwise, compute the thickness of the line and
                    # draw the connecting lines
                    thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
                    cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
                    
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor.predict_cli()


if __name__ == "__main__":
    predict()
