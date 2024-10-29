import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from train import extract, get_diffusion_params
from train import TIMESTEPS, IMAGE_SIZE, CHANNELS
from model import UNet

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def p_sample(model, x, t, params):
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
def sample_images(model, image_size, batch_size, channels, device, params):
    """Generate new images using the trained model"""
    x = torch.randn(batch_size, channels, image_size, image_size).to(device)

    for t in tqdm(reversed(range(TIMESTEPS)), desc="Sampling", total=TIMESTEPS):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        x = p_sample(model, x, t_batch, params)
        if t % 100 == 0:
            show_images(x)

        if x.isnan().any():
            raise ValueError(f"NaN detected in image at timestep {t}")

    return x


def show_images(images, title=""):
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


plt.figure(figsize=(10, 10))

params = get_diffusion_params(TIMESTEPS, device)

model = UNet(32, TIMESTEPS).to(device)
model.load_state_dict(torch.load("model.pkl", weights_only=True))

model.eval()
generated_images = sample_images(
    model=model,
    image_size=IMAGE_SIZE,
    batch_size=16,
    channels=CHANNELS,
    device=device,
    params=params,
)
show_images(generated_images, title="Generated Images")
plt.show()
