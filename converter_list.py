
import os
train_dir = 'C:/Users/T25301039/Documents/CVPR_Anti_UAV/3rd_Anti-UAV_train_val/train'
seqs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
with open(os.path.join(train_dir, 'list.txt'), 'w') as f:
    f.write('\n'.join(seqs))
print('Done! Written', len(seqs), 'sequences to list.txt')
