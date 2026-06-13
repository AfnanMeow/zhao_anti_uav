
import os,sys,torch,traceback,glob
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
CACHE_PATH=r'C:/Users/T25301039/.cache/modelscope/hub/damo/cv_tinynas_uav-detection_damoyolo'; WORK_DIR=r'C:/Users/T25301039/Documents/CVPR_Anti_UAV/fl_workdirs/client_ckpts\exp_c2_work'
GLOBAL_CKPT=r'C:/Users/T25301039/Documents/CVPR_Anti_UAV/fl_workdirs/client_ckpts\exp_c2_global.pt'; LOCAL_CKPT=r'C:/Users/T25301039/Documents/CVPR_Anti_UAV/fl_workdirs/client_ckpts\exp_c2_local.pt'
EPOCHS=5; BATCH_SIZE=8
_has_global=os.path.exists(GLOBAL_CKPT)
kwargs=dict(cfg_file=os.path.join(CACHE_PATH,'configuration.json'),
    model='damo/cv_tinynas_uav-detection_damoyolo',gpu_ids=[0],
    batch_size=BATCH_SIZE,max_epochs=EPOCHS,num_classes=1,
    cache_path=CACHE_PATH,load_pretrain=False,base_lr_per_img=0.00005,
    train_image_dir=r'C:/Users/T25301039/Documents/CVPR_Anti_UAV/3rd_Anti-UAV_train_val/train/20190925_130434_1_3',val_image_dir=r'C:/Users/T25301039/Documents/CVPR_Anti_UAV/3rd_Anti-UAV_train_val/train/20190925_130434_1_3',
    train_ann=r'C:/Users/T25301039/Documents/CVPR_Anti_UAV/single_test_coco/annotations/train.json',val_ann=r'C:/Users/T25301039/Documents/CVPR_Anti_UAV/single_test_coco/annotations/train.json',
    work_dir=WORK_DIR,num_workers=0)
if _has_global: kwargs['finetune_path']=GLOBAL_CKPT
try:
    trainer=build_trainer(name=Trainers.tinynas_damoyolo,default_args=kwargs)
    trainer.cfg.miscs.num_workers=0; trainer.cfg.miscs.print_interval_iters=20
    trainer.cfg.train.warmup_epochs=0
    trainer.cfg.train.warmup_start_lr=float(0.00005*BATCH_SIZE)
    trainer.cfg.train.augment.mosaic_mixup.mosaic_prob=0.0
    trainer.cfg.train.augment.mosaic_mixup.mixup_prob=0.0
    trainer.cfg.train.total_epochs=EPOCHS
    trainer.train()
    _cwd=os.getcwd()
    latest=os.path.join(_cwd,'workdirs','uav_detection','latest_ckpt.pth')
    epochN=os.path.join(_cwd,'workdirs','uav_detection',f'epoch_{EPOCHS}_ckpt.pth')
    best_ckpt=None
    for c in [latest,epochN]:
        if os.path.exists(c): best_ckpt=c; break
    if best_ckpt is None:
        found=glob.glob(os.path.join(_cwd,'workdirs','**','*.pth'),recursive=True)
        found=[f for f in found if 'latest' in f or 'epoch_' in f]
        if found: best_ckpt=max(found,key=os.path.getmtime)
    if best_ckpt:
        ckpt=torch.load(best_ckpt,map_location='cpu')
        if isinstance(ckpt,dict):
            if 'ema_state_dict' in ckpt: final_sd=ckpt['ema_state_dict']
            elif 'state_dict' in ckpt: final_sd=ckpt['state_dict']
            elif 'model' in ckpt: final_sd=ckpt['model']
            else: final_sd=ckpt
        else: final_sd=ckpt
        torch.save(final_sd,LOCAL_CKPT); print(f'[Client] Saved',flush=True)
    else: sys.exit(1)
except Exception as e: print(f'ERROR: {e}',flush=True); traceback.print_exc(); sys.exit(1)
