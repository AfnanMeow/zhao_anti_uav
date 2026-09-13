import os
import json
import glob
from PIL import Image

# Single folder path
seq_path = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/3rd_Anti-UAV_train_val/train/20190925_130434_1_3'
out_dir  = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/single_test_coco'

os.makedirs(os.path.join(out_dir, 'annotations'), exist_ok=True)

# Load labels
with open(os.path.join(seq_path, 'IR_label.json'), 'r') as f:
    label = json.load(f)

gt_rects    = label['gt_rect']
exist_flags = label['exist']

img_files = sorted(glob.glob(os.path.join(seq_path, '*.jpg')))
print(f'Found {len(img_files)} images, {sum(exist_flags)} frames with drone visible')

images      = []
annotations = []
ann_id      = 0

for img_id, (img_path, gt, exist) in enumerate(zip(img_files, gt_rects, exist_flags), start=1):
    try:
        with Image.open(img_path) as img:
            w, h = img.size
    except:
        w, h = 640, 512

    images.append({
        'id': img_id,
        'file_name': img_path.replace('\\', '/'),
        'width': w,
        'height': h
    })

    if exist and len(gt) == 4:
        x, y, bw, bh = gt
        if bw > 0 and bh > 0:
            ann_id += 1
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': 1,
                'bbox': [float(x), float(y), float(bw), float(bh)],
                'area': float(bw * bh),
                'iscrowd': 0
            })

# Split 80% train, 20% val
split = int(len(images) * 0.8)
train_imgs = images[:split]
val_imgs   = images[split:]
train_img_ids = {img['id'] for img in train_imgs}

train_anns = [a for a in annotations if a['image_id'] in train_img_ids]
val_anns   = [a for a in annotations if a['image_id'] not in train_img_ids]

coco_categories = [{'id': 1, 'name': 'uav', 'supercategory': 'object'}]

# Save train.json
with open(os.path.join(out_dir, 'annotations', 'train.json'), 'w') as f:
    json.dump({'images': train_imgs, 'annotations': train_anns, 'categories': coco_categories}, f)

# Save val.json
with open(os.path.join(out_dir, 'annotations', 'val.json'), 'w') as f:
    json.dump({'images': val_imgs, 'annotations': val_anns, 'categories': coco_categories}, f)

print(f'Train: {len(train_imgs)} images, {len(train_anns)} annotations')
print(f'Val:   {len(val_imgs)} images, {len(val_anns)} annotations')
print(f'Saved to {out_dir}/annotations/')