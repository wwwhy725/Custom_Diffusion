import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image

from model import Unet
from diffusion import DiffusionTrainer, DiffusionSampler

import os
import numpy as np
import copy
from tqdm import trange, tqdm
import math
from multiprocessing import cpu_count

from tensorboardX import SummaryWriter

def cycle(dl):
    while True:
        for data in dl:
            yield data

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"make directory {path} successfully!")
    else:
        print(f"{path} already exits!")

def closest_factors(a):
    factors = []
    for i in range(1, int(a ** 0.5) + 1):
        if a % i == 0:
            factors.append((i, a // i))
    closest_factor = min(factors, key=lambda x: abs(x[0] - x[1]))
    return closest_factor

def warmup_lr(step, warmup: int=5000):
    return min(step, warmup) / warmup

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))

# logging
logdir = './logs'
sample_dir = './logs/samples'
model_dir = './logs/model'
create_directory(sample_dir)
create_directory(model_dir)
writer = SummaryWriter(logdir)
writer.flush()

# dataset
class CustomDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.data = np.load(data_path)
        self.labels = np.load(labels_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_item = self.data[index]
        label_item = self.labels[index]
        return data_item, label_item
    
data_path = 'all_cifar_sorted.npy'
label_path = 'kmeans_labels_cifar.npy'
all_imgs = np.load(data_path)
all_labels = np.load(label_path)
all_labels_th = torch.tensor(all_labels)
mydata = CustomDataset(data_path=data_path, labels_path=label_path)
dataloader = DataLoader(mydata, batch_size=128, shuffle=False, pin_memory=True, num_workers=cpu_count()//2)
dl = cycle(dataloader)

# parameters
lr = 1e-4
grad_clip = 1.
ema_decay = 0.9999
total_steps = 100000
b, c, h, w = 16, 3, 32, 32
interval = 5000

# model set up
model = Unet(
    dim = 256,
    dim_mults = (1, 2, 4, 8),
    channels = 3,
    flash_attn = True
)
ema_model = copy.deepcopy(model)
optim = torch.optim.Adam(model.parameters(), lr=lr)
sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

# train
step = 0
with tqdm(total_steps, dynamic_ncols=True) as pbar:
    while step < total_steps:
        img, label = next(dl)
        unique_labels = torch.unique(label)
        find_label, _ = torch.max(torch.eq(all_labels_th, unique_labels[:, None]), dim=0)
        eps_dataset = all_imgs[find_label]

        # train
        optim.zero_grad()
        trainer = DiffusionTrainer(
            model=model,
            epsilon_star_data=eps_dataset,
        )
        loss = trainer(img).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip)
        optim.step()
        sched.step()
        ema(model, ema_model, ema_decay)

        # step
        step += 1

        # log
        writer.add_scalar('loss', loss, step)
        pbar.set_postfix(loss='%.3f' % loss)

        # sample every a few steps
        if step % interval == 0:
            diffusion_sampler = DiffusionSampler(model, sample_timesteps=1000)
            init_noise = torch.randn(b, c, h, w).to('cuda')
            samples = diffusion_sampler(init_noise)
            grid = make_grid((samples+1)/2, nrow=closest_factors(b))
            make_grid(grid, os.path.join(sample_dir, f'sample_{step//interval}.png'))

        # save model ever a few steps
        if step % interval == 0:
            torch.save(model.state_dict(), f'model_{step//interval}.pt')

        pbar.update(1)

