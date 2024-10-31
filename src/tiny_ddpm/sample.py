from typing import Dict, Union
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from train import extract, get_diffusion_params
from train import TIMESTEPS, IMAGE_SIZE, CHANNELS, DDIM_TIMESTEPS
from model import UNet

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def ddpm_sample(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Sample from the model at timestep t"""
    predicted_noise = model(x, t)

    one_over_alphas = extract(params["one_over_alphas"], t, x.shape)
    posterior_mean_coef = extract(params["posterior_mean_coef"], t, x.shape)

    pred_mean = one_over_alphas * (x - posterior_mean_coef * predicted_noise)

    posterior_variance = extract(params["posterior_variance"], t, x.shape)

    if t[0] > 0:
        noise = torch.randn_like(x)
        return pred_mean + torch.sqrt(posterior_variance) * noise
    else:
        return pred_mean


@torch.no_grad()
def ddim_sample(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Sample from the model in a non-markovian way (DDIM)"""
    stride = TIMESTEPS // DDIM_TIMESTEPS
    t_prev = t - stride
    predicted_noise = model(x, t)

    alphas_prod = extract(params["alphas_cumprod"], t, x.shape)
    valid_mask = (t_prev >= 0).view(-1, 1, 1, 1)
    safe_t_prev = torch.maximum(t_prev, torch.tensor(0, device=device))
    alphas_prod_prev = extract(params["alphas_cumprod"], safe_t_prev, x.shape)
    alphas_prod_prev = torch.where(
        valid_mask, alphas_prod_prev, torch.ones_like(alphas_prod_prev)
    )

    sigma = extract(params["ddim_sigma"], t, x.shape)

    pred_x0 = (x - (1 - alphas_prod).sqrt() * predicted_noise) / alphas_prod.sqrt()

    pred = (
        alphas_prod_prev.sqrt() * pred_x0
        + (1.0 - alphas_prod_prev).sqrt() * predicted_noise
    )

    if t[0] > 0:
        noise = torch.randn_like(x)
        pred = pred + noise * sigma

    return pred


@torch.no_grad()
def ddpm_sample_images(
    model: torch.nn.Module,
    image_size: int,
    batch_size: int,
    channels: int,
    device: torch.device,
    params: Dict[str, torch.Tensor],
):
    """Generate new images using the trained model"""
    x = torch.randn(batch_size, channels, image_size, image_size).to(device)

    for t in tqdm(reversed(range(TIMESTEPS)), desc="DDPM Sampling", total=TIMESTEPS):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        x = ddpm_sample(model, x, t_batch, params)
        if t % 100 == 0:
            show_images(x)

        if x.isnan().any():
            raise ValueError(f"NaN detected in image at timestep {t}")

    return x


def get_ddim_timesteps(
    total_timesteps: int, num_sampling_timesteps: int
) -> torch.Tensor:
    """Gets the timesteps used for the DDIM process."""
    assert total_timesteps % num_sampling_timesteps == 0
    stride = total_timesteps // num_sampling_timesteps
    timesteps = torch.arange(0, total_timesteps, stride)
    return timesteps.flip(0)


@torch.no_grad()
def ddim_sample_images(
    model: torch.nn.Module,
    image_size: int,
    batch_size: int,
    channels: int,
    device: torch.device,
    params: Dict[str, torch.Tensor],
):
    """Generate new images using the trained model"""
    x = torch.randn(batch_size, channels, image_size, image_size).to(device)

    timesteps = get_ddim_timesteps(TIMESTEPS, DDIM_TIMESTEPS)

    for i in tqdm(range(len(timesteps) - 1), desc="DDIM Sampling"):
        t = torch.full((batch_size,), timesteps[i], device=device, dtype=torch.long)
        x_before = x.clone()
        x = ddim_sample(model, x, t, params)

        if x.isnan().any():
            raise ValueError(f"NaN detected at timestep {timesteps[i]}")

        if i % 10 == 0:
            show_images(x)
    return x


def show_images(images: Union[torch.Tensor, np.array], title=""):
    """Display a batch of images in a grid"""
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    images = (images * 0.25 + 0.5).clip(0, 1)
    for idx in range(min(16, len(images))):
        plt.subplot(4, 4, idx + 1)
        plt.imshow(np.transpose(images[idx], (1, 2, 0)))
        plt.axis("off")
    plt.suptitle(title)
    plt.draw()
    plt.pause(0.001)


if __name__ == "__main__":
    plt.figure(figsize=(10, 10))

    params = get_diffusion_params(TIMESTEPS, device, eta=0.0)

    model = UNet(32, TIMESTEPS).to(device)
    model.load_state_dict(torch.load("model.pkl", weights_only=True))

    model.eval()
    generated_images = (
        ddpm_sample_images(  # change to ddim_sample_images here to enable DDIM
            model=model,
            image_size=IMAGE_SIZE,
            batch_size=16,
            channels=CHANNELS,
            device=device,
            params=params,
        )
    )
    show_images(generated_images, title="Generated Images")

    # Keep the plot open after generation is finished
    plt.show()
