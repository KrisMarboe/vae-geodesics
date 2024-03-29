import models.vae as v
import torch


def calculate_latents(model: v.EVAE, data_loader: torch.utils.data.DataLoader, device: torch.device) -> torch.Tensor:
    """Calculate the latent variables for a given dataset using the encoder network of the model.

    Args:
        model (v.EVAE): The model to use for encoding the data.
        data_loader (torch.utils.data.DataLoader): The data to encode.
        device (torch.device): The device to use for calculations.

    Returns:
        torch.Tensor: The latent variables of the data.
    """
    model.eval()
    latents = []
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            latents.append(model.encoder(x).mean)
    return torch.cat(latents, dim=0)
