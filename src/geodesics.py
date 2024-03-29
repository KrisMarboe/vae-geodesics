import models.vae as v
import models.flow as f
import numpy as np
import torch
from torch.distributions.kl import kl_divergence as KL
from tqdm import tqdm


def energy(z0, z1, curve_points: torch.Tensor, model: v.VAE | v.EVAE | f.Flow, n_samples=None):
    # Append z0 in the beginning and z1 in the end
    curve_points = torch.cat([z0[None], curve_points, z1[None]], dim=0)
    if isinstance(model, v.VAE):
        decoded_points_l = model.decoder(curve_points[:-1])
        decoded_points_r = model.decoder(curve_points[1:])
        E = KL(decoded_points_l, decoded_points_r).sum()
    elif isinstance(model, v.EVAE):
        if n_samples is None:
            n_samples = len(model.decoders)
        E = torch.zeros(len(curve_points)-1)
        for i in range(n_samples):
            decoded_points_l = np.random.choice(model.decoders)(curve_points[:-1])
            decoded_points_r = np.random.choice(model.decoders)(curve_points[1:])
            E += KL(decoded_points_l, decoded_points_r)
        E /= n_samples
        E = E.sum()
    elif isinstance(model, f.Flow):
        log_prob = torch.exp(model.log_prob(curve_points))
        weights = 2 / (log_prob[1:] + log_prob[:-1])
        E = (torch.pow(torch.norm(torch.diff(curve_points, dim=0), dim=1),2) * weights).sum()
    return E

def optimize_curve(z0, z1, curve_points, model: v.VAE | v.EVAE | f.Flow, optimizer, max_iter=1000, min_iter=100, threshold=10, n_samples=None):
    pbar = tqdm(range(max_iter))
    best_loss = float('inf')
    iterations_since_last_improvement = 0
    for i in pbar:
        optimizer.zero_grad()
        loss = energy(z0, z1, curve_points, model, n_samples)
        loss.backward()
        optimizer.step()
        if best_loss < loss:
            iterations_since_last_improvement += 1
            if i + 1 > min_iter and iterations_since_last_improvement > threshold:
                break
        else:
            iterations_since_last_improvement = 0
            best_loss = loss
        # Calculate difference in curve_points
        pbar.set_description(f"loss={loss:.3f}")
    return curve_points