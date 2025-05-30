# -*- coding: utf-8 -*-
"""minimal U-Net

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XlmuPIaiKLc9Z8pcuE7DunZcFevoCD-R
"""

# ─── Script B: Minimal U-Net (Skip=OFF, Double-Conv=ON) over first 1 050 patches ───

# 1️⃣ Mount & Imports
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

# 2️⃣ Hyperparams & Paths
BASE_PATH   = '/content/drive/MyDrive/95Clouds/38-Cloud_training'
MAX_SAMPLES = 1050
BATCH_SIZE  = 8
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR          = 1e-3
EPOCHS      = 1

# 3️⃣ Metrics
def dice(p,t,eps=1e-6):
    p=(p>0.5).float(); i=(p*t).sum()
    return (2*i+eps)/(p.sum()+t.sum()+eps)
def iou(p,t,eps=1e-6):
    p=(p>0.5).float(); i=(p*t).sum(); u=(p+t).clamp(0,1).sum()
    return (i+eps)/(u+eps)
def precision(p,t,eps=1e-6):
    p=(p>0.5).float(); tp=(p*t).sum(); fp=(p*(1-t)).sum()
    return (tp+eps)/(tp+fp+eps)
def recall(p,t,eps=1e-6):
    p=(p>0.5).float(); tp=(p*t).sum(); fn=((1-p)*t).sum()
    return (tp+eps)/(tp+fn+eps)
def pixel_acc(p,t):
    p=(p>0.5).float()
    return (p==t).float().sum()/torch.numel(t)

# 4️⃣ Dataset (RGB only, same file paths)
class CloudDataset(Dataset):
    def __init__(self, base_path):
        self.r_files  = sorted(glob.glob(os.path.join(base_path,'train_red'  ,'*.TIF')))
        self.g_files  = sorted(glob.glob(os.path.join(base_path,'train_green','*.TIF')))
        self.b_files  = sorted(glob.glob(os.path.join(base_path,'train_blue' ,'*.TIF')))
        self.gt_files = sorted(glob.glob(os.path.join(base_path,'train_gt'   ,'*.TIF')))
        assert len(self.r_files)==len(self.g_files)==len(self.b_files)==len(self.gt_files), \
               "File count mismatch in RGB/GT folders"
    def __len__(self):
        return len(self.r_files)
    def __getitem__(self, idx):
        def load(path):
            return np.array(Image.open(path),dtype=np.float32)/255.0
        r = load(self.r_files[idx])
        g = load(self.g_files[idx])
        b = load(self.b_files[idx])
        gt= (load(self.gt_files[idx])>0.5).astype(np.uint8)
        x = torch.tensor(np.stack([r,g,b],axis=0), dtype=torch.float32)
        y = torch.tensor(gt[None,:,:], dtype=torch.float32)
        return x, y

# 5️⃣ Wrap in Subset to first 1050
full_ds = CloudDataset(BASE_PATH)
sub_ds  = Subset(full_ds, list(range(min(MAX_SAMPLES, len(full_ds)))))
loader  = DataLoader(sub_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print(f"▶ Using {len(sub_ds)} / {len(full_ds)} patches")

# 6️⃣ Minimal U-Net (no skips, double conv everywhere)
class MinimalUNet(nn.Module):
    def __init__(self, in_ch=3, base_c=64):
        super().__init__()
        def C(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            )
        # encoder
        self.e1 = C(in_ch,   base_c)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = C(base_c,  base_c*2)
        self.p2 = nn.MaxPool2d(2)
        self.e3 = C(base_c*2,base_c*4)
        self.p3 = nn.MaxPool2d(2)
        self.e4 = C(base_c*4,base_c*8)
        self.p4 = nn.MaxPool2d(2)
        # bottleneck
        self.b  = C(base_c*8, base_c*8)
        # decoder (no concatenation)
        self.d4 = nn.Sequential(
            nn.ConvTranspose2d(base_c*8, base_c*4, 2,2), nn.ReLU(inplace=True),
            C(base_c*4, base_c*4)
        )
        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(base_c*4, base_c*2, 2,2), nn.ReLU(inplace=True),
            C(base_c*2, base_c*2)
        )
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(base_c*2, base_c,   2,2), nn.ReLU(inplace=True),
            C(base_c,   base_c)
        )
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(base_c,   base_c,   2,2), nn.ReLU(inplace=True),
            C(base_c,   base_c)
        )
        self.out = nn.Conv2d(base_c,1,1)

    def forward(self, x):
        x = self.e1(x); x = self.p1(x)
        x = self.e2(x); x = self.p2(x)
        x = self.e3(x); x = self.p3(x)
        x = self.e4(x); x = self.p4(x)
        x = self.b(x)
        x = self.d4(x)
        x = self.d3(x)
        x = self.d2(x)
        x = self.d1(x)
        return torch.sigmoid(self.out(x))

# 7️⃣ Instantiate, optimizer & loss
model   = MinimalUNet().to(DEVICE)
opt     = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCELoss()

# 8️⃣ Train for one epoch & accumulate metrics
sums = {'loss':0,'dice':0,'iou':0,'prec':0,'rec':0,'acc':0}
model.train()
for x,y in tqdm(loader, desc='Training 1 epoch'):
    x,y   = x.to(DEVICE), y.to(DEVICE)
    pred  = model(x)
    loss  = loss_fn(pred, y)
    opt.zero_grad(); loss.backward(); opt.step()

    bs = x.size(0)
    sums['loss'] += loss.item()*bs
    for fn,key in [(dice,'dice'),(iou,'iou'),
                   (precision,'prec'),(recall,'rec'),
                   (pixel_acc,'acc')]:
        for i in range(bs):
            sums[key] += fn(pred[i:i+1], y[i:i+1]).item()

# 9️⃣ Evaluate on same subset
model.eval()
with torch.no_grad():
    for x,y in tqdm(loader, desc='Evaluating'):
        x,y  = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        bs   = x.size(0)
        # no additional loss needed
        for fn,key in [(dice,'dice'),(iou,'iou'),
                       (precision,'prec'),(recall,'rec'),
                       (pixel_acc,'acc')]:
            for i in range(bs):
                sums[key] += fn(pred[i:i+1], y[i:i+1]).item()

# 🔟 Print metrics (average over train+eval: multiply EPOCHS+1)
N = len(sub_ds)*(EPOCHS+1)
print(f"\n— Minimal U-Net Metrics (train+eval) on {len(sub_ds)} patches —")
print(f"Loss:      {sums['loss']/len(sub_ds):.4f}")
print(f"Dice:      {sums['dice']/N:.4f}")
print(f"IoU:       {sums['iou']/N:.4f}")
print(f"Precision: {sums['prec']/N:.4f}")
print(f"Recall:    {sums['rec']/N:.4f}")
print(f"Accuracy:  {sums['acc']/N:.4f}")