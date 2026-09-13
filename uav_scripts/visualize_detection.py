import os
import json
import glob
import cv2
import torch
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Paths
val_img_dir  = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/3rd_Anti-UAV_train_val/train/20190925_130434_1_3'
val_ann_path = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/single_test_coco/annotations/val.json'
model_path   = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/workdirs/uav_detection/latest_ckpt.pth'
output_dir   = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/detection_viz'

os.makedirs(output_dir, exist_ok=True)

# Load model
uav_detection = pipeline(
    Tasks.domain_specific_object_detection,
    model='damo/cv_tinynas_uav-detection_damoyolo',
)
ckpt = torch.load(model_path, map_location='cpu')
uav_detection.model.load_state_dict(ckpt['state_dict'])
uav_detection.model.eval()

# Load val annotations
with open(val_ann_path, 'r') as f:
    val_data = json.load(f)

# Build image_id -> annotation lookup
ann_by_image = {}
for ann in val_data['annotations']:
    ann_by_image[ann['image_id']] = ann['bbox']  # [x, y, w, h]

print(f'Processing {len(val_data["images"])} images...')

for img_info in val_data['images']:
    img_id   = img_info['id']
    img_path = img_info['file_name']

    if not os.path.exists(img_path):
        continue

    frame = cv2.imread(img_path)
    if frame is None:
        continue

    # Draw ground truth box in GREEN
    if img_id in ann_by_image:
        x, y, w, h = [int(v) for v in ann_by_image[img_id]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'GT', (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Run detection and draw predicted box in RED
    output = uav_detection(img_path)
    if 'boxes' in output and len(output['boxes']) > 0:
        boxes  = output['boxes']
        scores = output['scores'] if 'scores' in output else [1.0] * len(boxes)

        for box, score in zip(boxes, scores):
            if float(score) < 0.3:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'Pred {score:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Save output image
    out_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_dir, out_name), frame)

print(f'\nDone! Images saved to: {output_dir}')
print('GREEN box = Ground Truth')
print('RED box   = Model Prediction')