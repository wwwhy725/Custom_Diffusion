import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image

from model import Unet
from diffusion import DiffusionTrainer, DiffusionSampler

import os
import numpy as np
import copy
from tqdm import trange, tqdm
from multiprocessing import cpu_count

from tensorboardX import SummaryWriter

# device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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

def warmup_lr(step, warmup:int=1000):
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
create_directory(logdir)
sample_dir = './logs/samples'
model_dir = './logs/model'
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
b, c, h, w = 16, 3, 32, 32  # for sample
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
    
data_path = '/mnt/store/lyx/github_projs/why/DDPM/all_cifar_sorted.npy'
label_path = '/mnt/store/lyx/github_projs/why/DDPM/kmeans_labels_cifar.npy'
all_imgs = np.load(data_path)
all_labels = np.load(label_path)
all_labels_th = torch.tensor(all_labels).to(device)
mydata = CustomDataset(data_path=data_path, labels_path=label_path)
dataloader = DataLoader(mydata, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=cpu_count()//2)
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
        img, label = next(dl)
        img = img.to(device)
        label = label.to(device)
        unique_labels = torch.unique(label)
        find_label, _ = torch.max(torch.eq(all_labels_th, unique_labels[:, None]), dim=0)
        eps_dataset = all_imgs[find_label.cpu().detach().numpy()]

        # train
        optim.zero_grad()
        trainer = DiffusionTrainer(
            model=model,
            epsilon_star_data=eps_dataset,
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
            save_image(grid, os.path.join(sample_dir, f'sample_{step//interval}.png'))
        # save model ever a few steps
        if step % interval == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f'model_{step//interval}.pt'))

        pbar.update(1)

