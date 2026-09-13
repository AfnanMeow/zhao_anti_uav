
import os
val_dir = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/3rd_Anti-UAV_train_val/validation'
seqs = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
with open(os.path.join(val_dir, 'list.txt'), 'w') as f:
    f.write('\n'.join(seqs))
print('Done! Written', len(seqs), 'sequences to list.txt')
