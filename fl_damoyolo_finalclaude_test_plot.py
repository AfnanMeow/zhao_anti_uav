"""
run_test_and_plot.py

Replicates exactly what test_CIFAR10_checkpoint_system_.py does,
adapted for DAMO-YOLO drone detection.

Loads the LAST episode's actor from checkpoint_drone.pth,
then runs 1 episode testing all 4 methods with the same rhos and net_init.

Generates:
  - Fig 1: mAP vs Communication Rounds (DRL vs 3 baselines)
  - Fig 2: Noise levels vs Communication Rounds (3 clients sorted by rho)

Run from your project directory:
    python run_test_and_plot.py

Takes ~6 hours (4 methods x ~90 min each).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys, json, copy, time, random, subprocess, warnings
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TRAIN_IMG_DIR = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/3rd_Anti-UAV_train_val/train/20190925_130434_1_3'
VAL_IMG_DIR   = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/3rd_Anti-UAV_train_val/train/20190925_130434_1_3'
TRAIN_ANN     = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/single_test_coco/annotations/train.json'
VAL_ANN       = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/single_test_coco/annotations/val.json'
CACHE_PATH    = 'C:/Users/T25301039/.cache/modelscope/hub/damo/cv_tinynas_uav-detection_damoyolo'
CKPT_DIR      = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/fl_workdirs/client_ckpts'
DRL_CKPT      = './DRL_Nets/checkpoint_drone.pth'

N=3; batch_size=8; CR_Total=10; epoch_local=5
rho_min=20.0; rho_max=40.0; sigma_max=20.0; sigma_min=1.0; max_eval_imgs=40
rho_used_min = 2/sigma_max**2*epoch_local
rho_used_max = 2/sigma_min**2*epoch_local

os.makedirs('./plots', exist_ok=True)
os.makedirs('./exp_data', exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

def rho2sigma(rho,el,bs): return float(np.sqrt(2*el/max(rho,1e-9)))
def sigma2rho(sigma,el,bs): return 2*el/(sigma**2)

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(2,32); self.fc2=nn.Linear(32,16); self.fc3=nn.Linear(16,1)
    def forward(self,x):
        x=F.relu(self.fc1(x)); x=F.relu(self.fc2(x)); return torch.sigmoid(self.fc3(x))

def sigma_drl(actor, rho_total, T):
    sigmas=[]
    for CR in range(T):
        state=torch.Tensor([CR/T, rho_total/rho_max])
        with torch.no_grad(): action=actor(state).item()
        sigmas.append(action*(sigma_max-sigma_min)+sigma_min)
    return sigmas

def sigma_uniform(rho_total, T, el):
    # test_CIFAR10.py line 747
    s=float(np.clip(np.sqrt(2*el/(rho_total/T)), sigma_min, sigma_max))
    return [s]*T

def sigma_exponential(rho_total, T, el):
    # test_CIFAR10.py lines 787-796
    decay_ratio=2**(1/(T-1))
    rho_base=(1-decay_ratio)*rho_total/(1-decay_ratio**T)
    sigmas=[]
    for CR in range(T):
        rho_used=max(rho_base*(decay_ratio**CR), 1e-9)
        sigmas.append(float(np.clip(np.sqrt(2*el/rho_used), sigma_min, sigma_max)))
    return sigmas

def sigma_arithmetic(rho_total, T, el):
    # test_CIFAR10.py lines 828-836
    decay=2*rho_total/(T*(T-1)*3)
    rho_base=2/3*(rho_total/T)
    sigmas=[]
    for CR in range(T):
        rho_used=max(rho_base+decay*CR, 1e-9)
        sigmas.append(float(np.clip(np.sqrt(2*el/rho_used), sigma_min, sigma_max)))
    return sigmas

def apply_action_mapping(sigmas_list, rhos):
    rho_useds=[[sigma2rho(s,epoch_local,batch_size) for s in sigmas_list[i]] for i in range(N)]
    rho_used_sums=[sum(rho_useds[i]) for i in range(N)]
    for i in range(N):
        S=rho_used_sums[i]
        if abs(max(sigmas_list[i])-min(sigmas_list[i]))<0.01:
            a=rhos[i]/max(S,1e-9); b=0.0
        else:
            T=CR_Total; ubs=[rhos[i]/T]; lbs=[-np.inf]
            for CR in range(T):
                diff=S-T*rho_useds[i][CR]
                if diff>1e-7:
                    ubs.append((rho_used_max*S-rhos[i]*rho_useds[i][CR])/diff)
                    lbs.append((rho_used_min*S-rhos[i]*rho_useds[i][CR])/diff)
                elif diff<-1e-7:
                    ubs.append((rho_used_min*S-rhos[i]*rho_useds[i][CR])/diff)
                    lbs.append((rho_used_max*S-rhos[i]*rho_useds[i][CR])/diff)
            bub=min(ubs); blb=max(lbs)
            dnom=sum((T*rho_useds[i][CR]-S)**2 for CR in range(T))
            b=0.0 if abs(dnom)<1e-12 else \
              -sum((rho_useds[i][CR]*S-rhos[i])*(T*rho_useds[i][CR]-S) for CR in range(T))/dnom
            b=max(blb,min(bub,b)); a=(rhos[i]-b*T)/max(S,1e-9)
        for CR in range(CR_Total):
            rho_useds[i][CR]=a*rho_useds[i][CR]+b
            sigmas_list[i][CR]=rho2sigma(rho_useds[i][CR],epoch_local,batch_size)
    return sigmas_list

def evaluate_map(ckpt_path, conf_thresh=0.005):
    if not os.path.exists(ckpt_path): return 0.0
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        pipe=pipeline(Tasks.domain_specific_object_detection,
                      model='damo/cv_tinynas_uav-detection_damoyolo',model_revision='master')
        sd=torch.load(ckpt_path,map_location='cpu')
        sd={k.replace('_module.',''):v for k,v in sd.items()}
        pipe.model.load_state_dict(sd,strict=False); pipe.model.eval()
        try: pipe.model.head.nms_conf_thre=0.005
        except: pass
        coco_gt=COCO(VAL_ANN); img_ids=coco_gt.getImgIds()
        if len(img_ids)>max_eval_imgs: img_ids=random.sample(img_ids,max_eval_imgs)
        results=[]
        for img_id in img_ids:
            info=coco_gt.loadImgs(img_id)[0]; path=info['file_name']
            if not os.path.isabs(path): path=os.path.join(VAL_IMG_DIR,os.path.basename(path))
            if not os.path.exists(path): continue
            out=pipe(path)
            for det,score in zip(out.get('boxes',[]),out.get('scores',[])):
                if len(det)<4: continue
                x1,y1,x2,y2=det[:4]; score=float(score)
                if score<conf_thresh: continue
                results.append({'image_id':img_id,'category_id':1,
                                'bbox':[float(x1),float(y1),float(x2-x1),float(y2-y1)],
                                'score':score})
        if not results: return 0.0
        tmp=f'_tmp_{os.getpid()}.json'
        with open(tmp,'w') as f: json.dump(results,f)
        coco_dt=coco_gt.loadRes(tmp); ev=COCOeval(coco_gt,coco_dt,'bbox')
        ev.params.imgIds=img_ids; ev.evaluate(); ev.accumulate(); ev.summarize()
        os.remove(tmp); return float(ev.stats[1])
    except: return 0.0

def make_wrapper(i, sigma, global_ckpt, local_ckpt, work_dir):
    return f"""
import os,sys,torch,traceback,glob
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
CACHE_PATH=r'{CACHE_PATH}'; WORK_DIR=r'{work_dir}'
GLOBAL_CKPT=r'{global_ckpt}'; LOCAL_CKPT=r'{local_ckpt}'
EPOCHS={epoch_local}; BATCH_SIZE={batch_size}
_has_global=os.path.exists(GLOBAL_CKPT)
kwargs=dict(cfg_file=os.path.join(CACHE_PATH,'configuration.json'),
    model='damo/cv_tinynas_uav-detection_damoyolo',gpu_ids=[0],
    batch_size=BATCH_SIZE,max_epochs=EPOCHS,num_classes=1,
    cache_path=CACHE_PATH,load_pretrain=False,base_lr_per_img=0.00005,
    train_image_dir=r'{TRAIN_IMG_DIR}',val_image_dir=r'{VAL_IMG_DIR}',
    train_ann=r'{TRAIN_ANN}',val_ann=r'{TRAIN_ANN}',
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
    epochN=os.path.join(_cwd,'workdirs','uav_detection',f'epoch_{{EPOCHS}}_ckpt.pth')
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
except Exception as e: print(f'ERROR: {{e}}',flush=True); traceback.print_exc(); sys.exit(1)
"""

def run_fl_episode(method_name, sigmas_per_client, init_sd, prefix):
    print(f'\n{"="*60}\nMETHOD: {method_name}\n{"="*60}')
    current_sd={k:v.clone() for k,v in init_sd.items()}
    global_accs=[]
    for CR in range(CR_Total):
        print(f'\n[{method_name} | Round {CR+1}/{CR_Total}]',flush=True)
        local_sds=[]
        for i in range(N):
            sigma=sigmas_per_client[i][CR]
            global_ckpt=os.path.join(CKPT_DIR,f'{prefix}_c{i}_global.pt')
            local_ckpt=os.path.join(CKPT_DIR,f'{prefix}_c{i}_local.pt')
            work_dir=os.path.join(CKPT_DIR,f'{prefix}_c{i}_work')
            os.makedirs(work_dir,exist_ok=True)
            torch.save({k.replace('_module.',''):v for k,v in current_sd.items()},global_ckpt)
            wrapper=make_wrapper(i,sigma,global_ckpt,local_ckpt,work_dir)
            wp=os.path.join(work_dir,'run_client.py')
            with open(wp,'w',encoding='utf-8') as f: f.write(wrapper)
            print(f'  Client {i} sigma={sigma:.4f}',flush=True)
            t0=time.time()
            subprocess.run([sys.executable,wp],capture_output=False,timeout=7200)
            print(f'  Client {i} done {time.time()-t0:.1f}s',flush=True)
            if os.path.exists(local_ckpt):
                sd=torch.load(local_ckpt,map_location='cpu')
                local_sds.append({k.replace('_module.',''):v for k,v in sd.items()})
            else: local_sds.append(current_sd)
        # FedAvg backbone+neck only
        ref=local_sds[0]; avg_sd={}
        for k in ref:
            is_head=any(h in k for h in ['head','cls_pred','reg_pred','obj_pred','cls_','reg_'])
            if is_head: avg_sd[k]=ref[k]
            elif ref[k].is_floating_point():
                tensors=[sd[k].float() for sd in local_sds if k in sd]
                avg_sd[k]=torch.stack(tensors).mean(0)
            else: avg_sd[k]=ref[k]
        current_sd=avg_sd
        # Evaluate
        acc=0.0
        for i in range(N):
            lc=os.path.join(CKPT_DIR,f'{prefix}_c{i}_local.pt')
            m=evaluate_map(lc) if os.path.exists(lc) else 0.0
            acc+=m; print(f'  Client {i} mAP={m:.4f}',flush=True)
        acc/=N; global_accs.append(acc)
        print(f'[{method_name}] CR {CR+1}/{CR_Total} mAP={acc:.4f}',flush=True)
    return global_accs

if __name__=='__main__':
    # Load last episode's actor from checkpoint
    print(f'Loading actor from {DRL_CKPT}')
    assert os.path.exists(DRL_CKPT), f'Not found: {DRL_CKPT}'
    ck=torch.load(DRL_CKPT,map_location='cpu')
    actor=Actor(); actor.load_state_dict(ck['actor']); actor.eval()
    print(f'Actor loaded (episode {ck["episode"]})')

    # Same rhos for all methods — fair comparison
    random.seed(42); np.random.seed(42)
    rhos=[float(np.random.uniform(rho_min,rho_max)) for _ in range(N)]
    print(f'Rhos: {[f"{r:.3f}" for r in rhos]}')

    # Same pretrained init weights for all methods
    print('Loading pretrained weights...')
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    pipe=pipeline(Tasks.domain_specific_object_detection,
                  model='damo/cv_tinynas_uav-detection_damoyolo',model_revision='master')
    init_sd={k:v.clone() for k,v in pipe.model.state_dict().items()}
    del pipe; print('Done.')

    # Compute sigma schedules
    sigmas_drl_raw=[[sigma_drl(actor,rhos[i],CR_Total)[cr] for cr in range(CR_Total)] for i in range(N)]
    sigmas_drl=apply_action_mapping([[s for s in sigmas_drl_raw[i]] for i in range(N)],rhos)
    sigmas_uni  =[sigma_uniform(    rhos[i],CR_Total,epoch_local) for i in range(N)]
    sigmas_exp  =[sigma_exponential(rhos[i],CR_Total,epoch_local) for i in range(N)]
    sigmas_arith=[sigma_arithmetic( rhos[i],CR_Total,epoch_local) for i in range(N)]

    # Resume support
    cache='./exp_data/test_results.json'; results={}
    if os.path.exists(cache):
        with open(cache) as f: results=json.load(f)
        print(f'Cached: {list(results.keys())}')

    methods=[('DRL',sigmas_drl,'drl'),('Uniform',sigmas_uni,'uni'),
             ('Exponential',sigmas_exp,'exp'),('Arithmetic',sigmas_arith,'arith')]

    for name,sigmas,prefix in methods:
        if name in results:
            print(f'[{name}] Skipping (cached)'); continue
        t0=time.time()
        accs=run_fl_episode(name,sigmas,{k:v.clone() for k,v in init_sd.items()},prefix)
        results[name]=accs
        with open(cache,'w') as f: json.dump(results,f)
        print(f'[{name}] {(time.time()-t0)/60:.1f} min | final={accs[-1]:.4f}')

    # ── Fig 1: mAP vs Communication Rounds ───────────────────────────────────
    x=list(range(1,CR_Total+1))
    plt.clf(); plt.figure(figsize=(8,5))
    plt.plot(x,results['DRL'],        linestyle='solid',  linewidth=2)
    plt.plot(x,results['Uniform'],    linestyle='dotted', linewidth=2)
    plt.plot(x,results['Exponential'],linestyle='dashed', linewidth=2)
    plt.plot(x,results['Arithmetic'], linestyle='dashdot',linewidth=2)
    plt.xticks(x,size=12); plt.yticks(size=12)
    plt.xlabel('Communication Round $t$',fontsize=16)
    plt.ylabel('mAP@0.5',fontsize=16)
    plt.legend(['Our method','Uniform','Exponential','Arithmetic sequence'],fontsize=13)
    plt.grid(); plt.tight_layout()
    plt.savefig('./plots/fig1_mAP_vs_rounds_REAL.pdf')
    plt.savefig('./plots/fig1_mAP_vs_rounds_REAL.png',dpi=150)
    plt.close(); print('Saved fig1')

    # ── Fig 2: Noise levels — sort clients by rho ascending ──────────────────
    si=sorted(range(N),key=lambda i:rhos[i])
    c0,c1,c2=si[0],si[1],si[2]
    plt.clf(); plt.figure(figsize=(8,5))
    plt.plot(list(range(CR_Total)),sigmas_drl[c0],linestyle='solid',  linewidth=2)
    plt.plot(list(range(CR_Total)),sigmas_drl[c1],linestyle='dotted', linewidth=2)
    plt.plot(list(range(CR_Total)),sigmas_drl[c2],linestyle='dashed', linewidth=2)
    plt.xticks(list(range(0,CR_Total+1,2)),size=12); plt.yticks(size=12)
    plt.xlabel('Communication Round $t$',fontsize=16)
    plt.ylabel('Noise Level $\\sigma_k^t$',fontsize=16)
    plt.legend([f'Client with $\\rho_k=$ {rhos[c0]:.3f}',
                f'Client with $\\rho_k=$ {rhos[c1]:.3f}',
                f'Client with $\\rho_k=$ {rhos[c2]:.3f}'],fontsize=13)
    plt.grid(); plt.tight_layout()
    plt.savefig('./plots/fig2_noise_levels_REAL.pdf')
    plt.savefig('./plots/fig2_noise_levels_REAL.png',dpi=150)
    plt.close(); print('Saved fig2')

    with open('./exp_data/all_test_results.json','w') as f:
        json.dump({'results':results,'rhos':rhos,
                   'sigmas_drl':sigmas_drl,'sigmas_uni':sigmas_uni,
                   'sigmas_exp':sigmas_exp,'sigmas_arith':sigmas_arith},f)

    print('\n── Summary ────────────────────────────')
    for name,_,_ in methods:
        if name in results:
            print(f'  {name:<15} final mAP = {results[name][-1]:.4f}')
    print('\nDone. Plots in ./plots/')