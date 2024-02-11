import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

import numpy as np
import os
import random
from multiprocessing import cpu_count

from model import Unet
from diffusion import DiffusionSampler
from fid_eval import FIDEvaluation

from utils import cycle, closest_factors, create_directory, CustomDataset

# seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# device
device = torch.device('cuda') if torch.cuda.is_available() == True else torch.device('cpu')

# setup
b, c, h, w = 25, 3, 64, 64
unet_dim = 256
model = Unet(
    dim = unet_dim,
    dim_mults = (1, 2, 4, 8),
    channels = 3,
    flash_attn = True
).to(device)

# directory
results_dir = './results'
create_directory(results_dir)
model_dir = './logs_celeba_std/model'

# load (pre)trained model
model_id = 20
model.load_state_dict(torch.load(os.path.join(model_dir, f'model_{model_id}.pt')))
model.eval()

# sampling
diffusion_sampler = DiffusionSampler(model, device, sample_timesteps=50)
# init_noise = torch.randn(b, c, h, w).to(device)
# samples = diffusion_sampler(init_noise)

# samples = (samples+1)/2
# # np.save('samples_cifar_17.npy', samples.cpu().detach().numpy())
# grid = make_grid(samples, nrow=closest_factors(b)[0])
# save_image(grid, os.path.join(results_dir, f'ddim.png'))

# fid
data_path = '/mnt/store/lyx/github_projs/why/DDPM/epsilon_star/celeba_10k/celeba_subset_10k.npy'
all_imgs = np.load(data_path)
mydata = CustomDataset(data_path=data_path)
dataloader = DataLoader(mydata, batch_size=b, shuffle=True, pin_memory=True, num_workers=cpu_count()//2)
dl = cycle(dataloader)

fid_scorer = FIDEvaluation(
    batch_size=b,
    dl=dl,
    sampler=diffusion_sampler,
    image_size=64,
    channels=3
)
fid = fid_scorer.fid_score()
