import os.path as osp
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
import os
import json

# Comment out MsDataset downloads — not needed
# train_dataset = MsDataset.load(...)
# val_dataset = MsDataset.load(...)

# Your local paths
train_img_dir   = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/3rd_Anti-UAV_train_val/train/20190925_130434_1_3'
val_img_dir     = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/3rd_Anti-UAV_train_val/train/20190925_130434_1_3'
train_anno_path = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/single_test_coco/annotations/train.json'
val_anno_path   = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/single_test_coco/annotations/val.json'
cache_path      = 'C:/Users/T25301039/.cache/modelscope/hub/damo/cv_tinynas_uav-detection_damoyolo'

kwargs = dict(
    cfg_file=os.path.join(cache_path, 'configuration.json'),
    model='damo/cv_tinynas_uav-detection_damoyolo',
    gpu_ids=[0],
    batch_size=8,
    max_epochs=50,
    num_classes=1,
    cache_path=cache_path,
    load_pretrain=False,   # train from scratch!
    base_lr_per_img=0.0001,
    train_image_dir=train_img_dir,
    val_image_dir=val_img_dir,
    train_ann=train_anno_path,
    val_ann=val_anno_path,
    work_dir='C:/Users/T25301039/Documents/CVPR_Anti_UAV/detection_checkpoints',
)

if __name__ == '__main__':
    trainer = build_trainer(
        name=Trainers.tinynas_damoyolo, default_args=kwargs)
    trainer.train()