import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float32)


class DiffusionTrainer(nn.Module):
    def __init__(
        self,
        model, 
        # epsilon_star_data: np.ndarray,
        device, 
        timesteps: int=1000
    ):
        super().__init__()
        self.model = model
        # self.eps_star_data = epsilon_star_data

        self.timesteps = timesteps
        self.device = device

        # betas and alphas
        self.register_buffer('betas', linear_beta_schedule(timesteps).to(self.device))

        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
    def epsilon_star_batch(self, x, t, data: np.ndarray, batch_size=1000):
        device = self.device
        b, c, h, w = x.shape  # take [256, 3, 32, 32] as an example
        num_train = data.shape[0]  # take 5k as an example
        num_batches = (num_train + batch_size - 1) // batch_size

        coef_deno = 1. / (-2 * (1. - self.alphas_cumprod))
        coef_x = 1. / torch.sqrt(1. - self.alphas_cumprod)
        coef_x_hat = self.sqrt_alphas_cumprod / torch.sqrt(1. - self.alphas_cumprod)
        # *********************************
        numerator = torch.zeros(b, c, h, w).to(device)  # [256, 3, 32, 32]
        denominator = torch.zeros(b).to(device)
        # deal with batch
        max_norm = -torch.inf * torch.ones(b).to(device)
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_train)
            train_data = torch.from_numpy(data[start_idx:end_idx]).to(device)  # a batch of train data  [batch_size, 3, 32, 32]
            batch_train_data = train_data.repeat(b, 1, 1, 1).reshape(b, train_data.shape[0], c, h, w)  # [256, batch_size, 3, 32, 32]
            x_minus_x0 = x[:, None, :, :, :] - extract(self.sqrt_alphas_cumprod, t, batch_train_data.shape) * batch_train_data  # [256, batch_size, 3, 32, 32]
            norm = x_minus_x0.norm(p=2, dim=(2, 3, 4)) ** 2   # [256, batch_size]
            norm = extract(coef_deno, t, norm.shape) * norm
            # max of norm
            max_norm = torch.max(max_norm, norm.max(dim=1)[0])

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_train)
            train_data = torch.from_numpy(data[start_idx:end_idx]).to(device)  # a batch of train data  [batch_size, 3, 32, 32]
            batch_train_data = train_data.repeat(b, 1, 1, 1).reshape(b, train_data.shape[0], c, h, w)  # [256, batch_size, 3, 32, 32]
            x_minus_x0 = x[:, None, :, :, :] - extract(self.sqrt_alphas_cumprod, t, batch_train_data.shape) * batch_train_data  # [256, batch_size, 3, 32, 32]
            norm = x_minus_x0.norm(p=2, dim=(2, 3, 4)) ** 2   # [256, batch_size]
            norm = extract(coef_deno, t, norm.shape) * norm
            # softmax
            # numerator -- a tensor: \sum [exp()*xi]
            numerator += (torch.exp(norm - max_norm[:, None])[:, :, None, None, None] * train_data[None, :, :, :, :]).sum(dim=1)  # [256, 3, 32, 32]

            # denominator -- a real number \sum exp()
            denominator += torch.sum(torch.exp(norm - max_norm[:, None]), dim=1)  # [256, ]
    
        weighted_x = numerator / denominator[:, None, None, None]    
        result = extract(coef_x, t, x.shape) * x - extract(coef_x_hat, t, weighted_x.shape) * weighted_x

        return result
        
    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.timesteps, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_cumprod, t, noise.shape) * noise
            )
        
        target = noise
        # target = self.epsilon_star_batch(x_t, t, self.eps_star_data)  # usually the target is noise, but here is eps_star

        loss = F.mse_loss(self.model(x_t, t), target, reduction='none')
        
        return loss

class DiffusionSampler(nn.Module):
    def __init__(self, model, device, sample_timesteps:int=1000, timesteps:int=1000):
        super().__init__()
        self.model = model
        self.device = device
        self.timesteps = timesteps
        self.sample_time_step = sample_timesteps
        if sample_timesteps < timesteps:
            self.sample_method = 'ddim'
        else:
            self.sample_method = 'ddpm'

        # betas and alphas
        betas = linear_beta_schedule(timesteps).to(self.device)
        self.register_buffer('betas', betas)
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)     
        
        # ddpm coeff
        if self.sample_method == 'ddpm':
            self.register_buffer('sigma', torch.sqrt(self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)))
            self.register_buffer('eps_coeff', betas / torch.sqrt(alphas*(1. - alphas_cumprod)))
            self.register_buffer('x_coeff', 1. / torch.sqrt(alphas))
        # ddim coeff
        elif self.sample_method == 'ddim':
            self.register_buffer('alphas_cumprod', alphas_cumprod)   


    def forward(self, x_T):
        """
        Algorithm 2.
        """
        b, c, h, w = x_T.shape
        x_t = x_T
        model = self.model
        model.eval()
        with torch.no_grad():
            if self.sample_method == 'ddpm':
                for time_step in tqdm(reversed(range(self.timesteps)), desc = 'sampling loop time step', total = self.sample_time_step):  # tbd
                    t = torch.full((b,), time_step, device=self.device)
                    eps = model(x_t, t)
                    z = torch.randn_like(x_t)
                    gamma = extract(self.eps_coeff, t, x_t.shape)*eps - extract(self.sigma, t, z.shape)*z
                    x_t = extract(self.x_coeff, t, x_t.shape)*x_t - gamma
                x_0 = x_t
            elif self.sample_method == 'ddim':
                times = torch.linspace(-1, self.timesteps - 1, steps = self.sample_time_step + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
                times = list(reversed(times.int().tolist()))
                time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
                for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):  # tbd
                    alpha = self.alphas_cumprod[time]
                    t = torch.full((b,), time, device=self.device)
                    eps = model(x_t, t)
                    if time_next < 0:
                        x_t = (x_t - (1. - alpha).sqrt() * eps) / alpha.sqrt()
                        continue
                    alpha_next = self.alphas_cumprod[time_next]
                    eps_coeff = ((1. - alpha_next) / alpha_next).sqrt() - ((1. - alpha) / alpha).sqrt()
                    x_t = alpha_next.sqrt() * (x_t/(alpha.sqrt()) + eps_coeff * eps)
                    
                x_0 = x_t
            else:
                raise NotImplementedError(self.sample_method)

        return torch.clip(x_0, -1, 1)