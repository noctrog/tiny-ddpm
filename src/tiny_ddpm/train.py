from typing import Dict, Callable
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import UNet

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
NUM_EPOCHS = 256
BATCH_SIZE = 128
IMAGE_SIZE = 32
CHANNELS = 3
TIMESTEPS = 1000
DDIM_TIMESTEPS = 100

NORM_MEAN = (0.4914, 0.4822, 0.4465)
NORM_STD = (0.2470, 0.2435, 0.2616)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ]
)

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True
)


def get_diffusion_params(
    timesteps: int,
    device: torch.device,
    ddim_timesteps: int = DDIM_TIMESTEPS,
    eta=0.0,
) -> Dict[str, torch.Tensor]:
    def linear_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    betas = linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    one_over_alphas = 1.0 / torch.sqrt(alphas)
    posterior_mean_coef = betas / sqrt_one_minus_alphas_cumprod

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    ddim_sigma = eta * torch.sqrt(
        (1.0 - alphas_cumprod_prev)
        / (1.0 - alphas_cumprod)
        * (1 - alphas_cumprod / alphas_cumprod_prev)
    )

    return {
        # DDPM Parameters
        "betas": betas.to(device),
        "alphas_cumprod": alphas_cumprod.to(device),
        "posterior_variance": posterior_variance.to(device),
        "one_over_alphas": one_over_alphas.to(device),
        "posterior_mean_coef": posterior_mean_coef.to(device),
        # DDIM Parameters
        "ddim_sigma": ddim_sigma.to(device),
    }


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Tensor.shape):
    """Extract coefficients at specified timesteps t"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def get_loss_fn(model: torch.nn.Module, params: Dict[str, torch.Tensor]) -> Callable:
    def loss_fn(x_0):
        batch_size = x_0.shape[0]
        t = torch.randint(0, TIMESTEPS, (batch_size,), device=device)
        noise = torch.randn_like(x_0)

        # Get the noisy image
        alpha_cumprod = extract(params["alphas_cumprod"], t, x_0.shape)
        noise_level = torch.sqrt(1.0 - alpha_cumprod)
        x_noisy = torch.sqrt(alpha_cumprod) * x_0 + noise_level * noise

        # Get predicted noise
        predicted_noise = model(x_noisy, t)

        return torch.nn.functional.mse_loss(predicted_noise, noise)

    return loss_fn


def train_epoch(
    model: torch.nn.Module, optimize, train_loader: DataLoader, loss_fn: Callable
) -> float:
    model.train()
    total_loss = 0

    with tqdm(train_loader, leave=False) as pbar:
        for batch in pbar:
            images = batch[0].to(device)
            optimizer.zero_grad()

            loss = loss_fn(images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_description(f"Loss: {loss.item():.4f}")

    return total_loss / len(train_loader)


if __name__ == "__main__":
    model = UNet(32, TIMESTEPS).to(device)
    nb_params = count_parameters(model)
    print(f"Total number of parameters: {nb_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    params = get_diffusion_params(TIMESTEPS, device)
    loss_fn = get_loss_fn(model, params)

    # Main training loop
    for e in tqdm(range(NUM_EPOCHS)):
        train_epoch(model, optimizer, train_loader, loss_fn)

    # Save model after training
    torch.save(model.state_dict(), "model.pkl")
