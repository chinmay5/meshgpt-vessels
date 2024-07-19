import argparse
import os
import random
import shutil
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import nn
from tqdm import tqdm

from environment_setup import PROJECT_ROOT_DIR
from mesh_on_vessels.diffusion_models.utils.denoising_model import SimpleLatentDenoiser


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, latents, masks):
        super(LatentDataset, self).__init__()
        self.latents = latents
        self.masks = masks


    def __len__(self):
        return len(self.latents)

    def __getitem__(self, item):
        return self.latents[item], self.masks[item]

    @property
    def hidden_dim(self):
        return self.latents.size(-1)


class FlowMatching(torch.nn.Module):
    """x0: noise, x1: data"""

    def __init__(
            self,
            model: torch.nn.Module,
            use_lognorm_is: bool = True,  # lognorm time importance sampling from SD3
            lognorm_mu: float = 0.0,
            lognorm_sigma: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.use_lognorm_is = use_lognorm_is
        self.lognorm_mu = lognorm_mu
        self.lognorm_sigma = lognorm_sigma

    def compute_loss(
            self, data_samples: torch.Tensor,
            mask: torch.Tensor,
            cond: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute loss for training."""
        x_1 = data_samples
        x_0 = torch.randn_like(x_1)
        v_gt = x_1 - x_0  # gt velocity is time-constant given x0 and x1

        if self.use_lognorm_is:
            # Apply lognorm SD3 importance sampling here
            t = torch.randn(x_1.shape[0], device=x_1.device, dtype=x_1.dtype)
            t = t * self.lognorm_sigma + self.lognorm_mu
            t = torch.sigmoid(t)
        else:
            t = torch.rand(x_1.shape[0], device=x_1.device, dtype=x_1.dtype)

        t_unsq = t.reshape(t.shape[0], *((1,) * (x_1.ndim - 1)))
        x_t = torch.lerp(x_0, x_1, t_unsq)

        v_pred = self.model(x_t, t, mask)
        loss_unreduced = torch.nn.functional.mse_loss(v_pred[mask], v_gt[mask], reduction="none")
        loss_unreduced = loss_unreduced.flatten(1).mean(1)
        loss = loss_unreduced.mean()

        return loss

    @torch.inference_mode()
    def sample(
            self,
            noise: torch.Tensor,
            mask: torch.Tensor,
            cond: torch.Tensor = None,  # [1, N, 3]
            num_steps: int = 100,
            reverse_time: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generation: discretize the ODE via Euler integration."""
        num_samples = noise.shape[0]
        x_t = noise
        traj = [x_t.detach().clone()]
        dt = 1.0 / num_steps
        times = [i / num_steps for i in range(num_steps)]
        if reverse_time:
            dt *= -1
            times = [1.0 - t for t in times]
        for i, t in enumerate(times):
            t_tnsr = torch.full([num_samples], t, dtype=x_t.dtype, device=x_t.device)
            v = self.model(x_t, t_tnsr, mask)
            x_t = x_t + (v * dt)
            traj.append(x_t.detach().clone())
        traj = torch.stack(traj, 1)  # num-shapes x num-timepoints x d
        return x_t, traj


# Let us also define the Diffusion model
class Diffusion:
    def __init__(self, model: torch.nn.Module, noise_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_pos(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))


    def compute_loss(
            self,
            data_samples: torch.Tensor,
            mask: torch.Tensor,
            cond: torch.Tensor = None
            ) -> torch.Tensor:
        t = self.sample_timesteps(data_samples.shape[0]).to(data_samples.device)
        x_t, noise = self.noise_pos(data_samples, t)
        predicted_noise = self.model(x_t, t, mask)
        mse_loss = torch.nn.functional.mse_loss(noise * mask.unsqueeze(-1), predicted_noise * mask.unsqueeze(-1))
        return mse_loss


    def sample(self,
            noise: torch.Tensor,
            mask: torch.Tensor):
        self.model.eval()
        x = noise
        with torch.no_grad():
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(x.shape[0]) * i).long().to(x.device)
                predicted_noise = self.model(x, t, mask)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        self.model.train()
        return x, None



def train_one_epoch(train_loader, flow_matching, optimizer, device):
    flow_matching.model.train()
    loss = 0
    for data, masks in tqdm(train_loader):
        data, masks = data.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = flow_matching.compute_loss(data, masks)
        # Autocasting efforts done
        loss.backward()
        torch.nn.utils.clip_grad_value_(flow_matching.model.parameters(), 1)
        optimizer.step()
    return float(loss)


@torch.inference_mode
def sample_latents(flow_matching, device, save_dir, save_latents=True, num_samples=100, latent_dims=512, val_loader=None):
    print("Sampling latents for Mesh GPT...")
    flow_matching.model.eval()
    generated_sample_tensor = []
    # NOTE: There are some issues based on padding. However, the hope is that since only the last dimension
    # is padded, we already get some meaningful idea of the distribution.
    if val_loader is not None:
        for data, mask in val_loader:
            noise = torch.randn_like(data)
            noise, mask = noise.to(device), mask.to(device)
            generated_sample, _ = flow_matching.sample(noise=noise, mask=mask)
            generated_sample_tensor.append(generated_sample.cpu())
        generated_sample_tensor = torch.cat(generated_sample_tensor, dim=0)
    else:
        noise = torch.randn((num_samples, latent_dims), device=device)
        # NOTE: 800 is the number of faces we used initially. Hence, the value is hard-coded. Perhaps, we can improve it
        mask = torch.ones((num_samples, 800), dtype=torch.bool).to(device)
        generated_sample_tensor, _ = flow_matching.sample(noise=noise, mask=mask)
    flow_matching.model.train()
    if save_latents:
        # Let us just save all the obtained latents.
        # We save the results as a normal tensor.
        torch.save(generated_sample_tensor, f"{save_dir}/sampled_latents.pt")
        return
    return generated_sample_tensor


def create_model(args, latent_dims):
    model = SimpleLatentDenoiser(latent_dim=latent_dims, hidden_dim=args.d_model, n_layers=args.num_layers, eps=1e-5)
    return model


def plot_tsne_curve(sampled_latents, val_loader, epoch):
    """
    Computes the tsne plot. Code taken from ChatGPT.
    :param sampled_latents: The latents sampled by the flow matching model
    :param val_loader: The dataloader for the validation set
    :param epoch: The current epoch for which plots are generated
    :return: None
    """
    plot_save_dir = f'{PROJECT_ROOT_DIR}/mesh_on_vessels/diffusion_models/tsne_plots'
    os.makedirs(plot_save_dir, exist_ok=True)
    # Load the whole val set at one go
    val_data = []
    for val_minibatch, _ in val_loader:
        val_data.append(val_minibatch)
    val_data = torch.cat(val_data, dim=0)
    num_samples = sampled_latents.shape[0]
    # Reshape the latents since things should be N, D
    sampled_latents = sampled_latents.view(num_samples, -1)
    val_data = val_data.view(num_samples, -1)
    # Convert tensors to NumPy arrays
    sampled_latents_numpy, val_data_numpy = sampled_latents.numpy(), val_data.numpy()
    # Create labels
    labels1 = np.zeros(num_samples)  # Label 0 for training data
    labels2 = np.ones(num_samples)  # Label 1 for test data

    # Concatenate the data and labels
    data = np.concatenate((sampled_latents_numpy, val_data_numpy), axis=0)
    labels = np.concatenate((labels1, labels2), axis=0)

    # Apply t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(data)

    # Plotting the t-SNE results
    plt.figure(figsize=(10, 5))

    # Scatter plot for each class
    plt.scatter(data_tsne[labels == 0, 0], data_tsne[labels == 0, 1], label='Sampled', alpha=0.5)
    plt.scatter(data_tsne[labels == 1, 0], data_tsne[labels == 1, 1], label='Val data', alpha=0.5)

    plt.legend()
    plt.title('t-SNE Plot')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.savefig(f"{plot_save_dir}/tsne_{epoch}.png")
    plt.show()
    del val_data


def train_flow_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = create_dataloaders()
    latent_dims = train_loader.dataset.hidden_dim
    # Create the model
    model = create_model(args, latent_dims=latent_dims)
    if args.model == 'flow':
        flow_matching = FlowMatching(model=model)
    else:
        flow_matching = Diffusion(model=model)
    model = model.to(device)
    save_dir = f"{PROJECT_ROOT_DIR}/mesh_on_vessels/diffusion_models/infer_samples"
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir)
    checkpoint_dir = f"{PROJECT_ROOT_DIR}/mesh_on_vessels/diffusion_models/infer_samples/checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("Starting training")
    if args.train:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
        times = []
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            loss = train_one_epoch(train_loader, flow_matching, optimizer, device)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            times.append(time.time() - start)
            if epoch % args.val_every == (args.val_every - 1):
                # We compare the sampled output and the validation set latents using a tsne plot
                sampled_latents = sample_latents(flow_matching, device, save_dir,
                                                 save_latents=False, num_samples=-1,
                                                 latent_dims=latent_dims, val_loader=val_loader)
                plot_tsne_curve(sampled_latents, val_loader, epoch)
        print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
        # Now we will sample based on the obtained model
        sample_latents(flow_matching, device, save_dir, save_latents=True, num_samples=args.num_gen_samples,
                                                 latent_dims=latent_dims)
        torch.save(model.state_dict(), f"{checkpoint_dir}/last_model.pth")
    else:
        print("Only sampling mode activated")
        # Now we will sample based on the obtained model
        print(model.load_state_dict(torch.load(f"{checkpoint_dir}/last_model.pth")))
        sample_latents(flow_matching, device, save_dir, save_latents=True, num_samples=args.num_gen_samples,
                                                 latent_dims=latent_dims)


def create_dataloaders():
    data_dict =  torch.load(f'{PROJECT_ROOT_DIR}/mesh_on_vessels/outputs/cont_recons_tensor.pt')
    all_latents, all_masks = data_dict['data'], data_dict['mask']
    if not all([os.path.exists(f'{PROJECT_ROOT_DIR}/mesh_on_vessels/diffusion_models/train_indices.npy'),
                os.path.exists(f'{PROJECT_ROOT_DIR}/mesh_on_vessels/diffusion_models/val_indices.npy')
                ]):
        all_indices = list(range(all_latents.size(0)))
        # Shuffle the indices
        random.shuffle(all_indices)
        train_index_length = int(0.9 * len(all_indices))
        train_indices, val_indices = all_indices[:train_index_length], all_indices[train_index_length:]
        assert len(set(train_indices).intersection(set(val_indices))) == 0, "Indices overlap. Please check"
        np.save(f'{PROJECT_ROOT_DIR}/mesh_on_vessels/diffusion_models/train_indices.npy', train_indices)
        np.save(f'{PROJECT_ROOT_DIR}/mesh_on_vessels/diffusion_models/val_indices.npy', val_indices)
    else:
        train_indices = np.load(f'{PROJECT_ROOT_DIR}/mesh_on_vessels/diffusion_models/train_indices.npy')
        val_indices = np.load(f'{PROJECT_ROOT_DIR}/mesh_on_vessels/diffusion_models/val_indices.npy')
    train_dataset = LatentDataset(all_latents[train_indices], all_masks[train_indices])
    val_dataset = LatentDataset(all_latents[val_indices], all_masks[val_indices])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8)
    return train_loader, val_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--val_every', type=int, default=100)
    parser.add_argument('--num_gen_samples', type=int, default=100)
    parser.add_argument('--model', type=str, default='diff')
    args = parser.parse_args()
    train_flow_model(args)
