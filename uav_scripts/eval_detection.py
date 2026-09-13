import os
import json
import glob
import cv2
import numpy as np
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Your paths
val_img_dir   = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/3rd_Anti-UAV_train_val/train/20190925_130434_1_3'
val_ann_path  = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/single_test_coco/annotations/val.json'
model_path    = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/workdirs/uav_detection/latest_ckpt.pth'

# Load pipeline with your trained model
uav_detection = pipeline(
    Tasks.domain_specific_object_detection,
    model='damo/cv_tinynas_uav-detection_damoyolo',
    model_revision='master'
)

# Load your trained weights
uav_detection.model.load_state_dict(
    torch.load(model_path, map_location='cpu')['state_dict']
)
uav_detection.model.eval()

# Load val annotations
coco_gt = COCO(val_ann_path)
image_ids = coco_gt.getImgIds()

results = []

print(f'Running detection on {len(image_ids)} validation images...')

for i, img_id in enumerate(image_ids):
    img_info = coco_gt.loadImgs(img_id)[0]
    img_path = img_info['file_name']

    if not os.path.exists(img_path):
        continue

    # Run detection
    frame = cv2.imread(img_path)
    if frame is None:
        continue

    output = uav_detection(img_path)

    if 'boxes' in output and len(output['boxes']) > 0:
        boxes = output['boxes']
        scores = output['scores'] if 'scores' in output else [1.0] * len(boxes)

        for box, score in zip(boxes, scores):
        # Only keep high confidence detections
            if float(score) < 0.3:
                continue
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            results.append({
                'image_id': img_id,
                'category_id': 1,
                'bbox': [float(x1), float(y1), float(w), float(h)],
                'score': float(score)
            })

    if (i + 1) % 20 == 0:
        print(f'Processed {i+1}/{len(image_ids)} images')

print(f'Total detections: {len(results)}')

if len(results) == 0:
    print('No detections found! Model may not be detecting anything.')
else:
    # Save results
    results_file = 'detection_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f)

    # Run COCO evaluation
    coco_dt = coco_gt.loadRes(results_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print('\n========== DETECTION RESULTS ==========')
    print(f'mAP@0.5:0.95 = {coco_eval.stats[0]:.4f}')
    print(f'mAP@0.5      = {coco_eval.stats[1]:.4f}')  # ← YOUR MAIN SCORE
    print(f'mAP@0.75     = {coco_eval.stats[2]:.4f}')
    print('=======================================')