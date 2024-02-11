import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image

from model import Unet
from diffusion import DiffusionTrainer, DiffusionSampler

import os
import numpy as np
import copy
import random
from tqdm import trange, tqdm
from multiprocessing import cpu_count

from utils import cycle, create_directory, closest_factors, ema, CustomDataset

from tensorboardX import SummaryWriter

# seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def warmup_lr(step, warmup:int=1000):
    return min(step, warmup) / warmup

# logging
logdir = './logs_celeba_std'
create_directory(logdir)
sample_dir = './logs_celeba_std/samples'
model_dir = './logs_celeba_std/model'
create_directory(sample_dir)
create_directory(model_dir)
writer = SummaryWriter(logdir)
writer.flush()

# parameters
batch_size = 128
lr = 1e-4
grad_clip = 1.
ema_decay = 0.9999
total_steps = 100000
b, c, h, w = 16, 3, 64, 64  # for sample
interval = 5000
unet_dim = 256
writer.add_scalar('hyperparameters/batch_size', batch_size, 0)
writer.add_scalar('hyperparameters/lr', lr, 0)
writer.add_scalar('hyperparameters/grad_clip', grad_clip, 0)
writer.add_scalar('hyperparameters/ema_decay', ema_decay, 0)
writer.add_scalar('hyperparameters/total_steps', total_steps, 0)
writer.add_scalar('hyperparameters/unet_dim', unet_dim, 0)
writer.add_scalar('hyperparameters/warmup', 1000, 0)

# dataset
# data_path = '/mnt/store/lyx/github_projs/why/DDPM/cifar_10k_sorted.npy'  # ori
data_path = '/mnt/store/lyx/github_projs/why/DDPM/epsilon_star/celeba_10k/celeba_subset_10k.npy'
label_path = '/mnt/store/lyx/github_projs/why/DDPM/kmeans_labels_10k.npy'
all_imgs = np.load(data_path)
# all_labels = np.load(label_path)  # ori
# all_labels_th = torch.tensor(all_labels).to(device)  # ori
# mydata = CustomDataset(data_path=data_path, labels_path=label_path)  # ori
mydata = CustomDataset(data_path=data_path)
dataloader = DataLoader(mydata, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count()//2)
dl = cycle(dataloader)

# model set up
model = Unet(
    dim = unet_dim,
    dim_mults = (1, 2, 4, 8),
    channels = 3,
    flash_attn = True
).to(device)
ema_model = copy.deepcopy(model)
optim = torch.optim.Adam(model.parameters(), lr=lr)
sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

# train
step = 0
with tqdm(initial = step, total = total_steps, dynamic_ncols=True) as pbar:
    while step < total_steps:
        # img, label = next(dl)
        # img = img.to(device)
        # label = label.to(device)
        # unique_labels = torch.unique(label)
        # find_label, _ = torch.max(torch.eq(all_labels_th, unique_labels[:, None]), dim=0)
        # eps_dataset = all_imgs[find_label.cpu().detach().numpy()]

        img = next(dl)
        img = img.to(device)
        # rand_ind = np.random.permutation(10000)[:500]  # random sample 1000 imgs
        # eps_dataset = all_imgs[rand_ind]
        # eps_dataset = np.concatenate((eps_dataset, img.cpu().detach().numpy()), axis=0)

        # train
        optim.zero_grad()
        trainer = DiffusionTrainer(
            model=model,
            # epsilon_star_data=eps_dataset,
            device=device
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
        pbar.set_description(f'loss: {loss:.4f}')

        # sample every a few steps
        if step % interval == 0:
            diffusion_sampler = DiffusionSampler(model, device, sample_timesteps=1000)
            init_noise = torch.randn(b, c, h, w).to(device)
            samples = diffusion_sampler(init_noise)
            grid = make_grid((samples+1)/2, nrow=closest_factors(b)[0])
            writer.add_image('sample', grid, step)
            save_image(grid, os.path.join(sample_dir, f'sample_{step//interval}.png'))
        # save model ever a few steps
        if step % interval == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f'model_{step//interval}.pt'))

        pbar.update(1)
writer.close()

