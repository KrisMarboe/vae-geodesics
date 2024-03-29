from data.mnist import load_mnist
from utils import calculate_latents
import models.flow as f
import models.vae as v
import os

import torch
import torch.nn as nn
import numpy as np
import yaml
from argparse import ArgumentParser

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('mode', type=str, choices=["vae", "flow", "both"], default="both", help="Model to train (default: %(default)s)")
    parser.add_argument('--model-dir', type=str, default='models/vae_mnist', help="Directory to save models (default: %(default)s)")
    parser.add_argument('--vae-name', type=str, default='vae', help="Name of the VAE model (default: %(default)s)")
    parser.add_argument('--flow-name', type=str, default='flow', help="Name of the Flow model (default: %(default)s)")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (default: %(default)s)")
    parser.add_argument('--vae-epochs', type=int, default=200, help="Number of epochs for VAE (default: %(default)s)")
    parser.add_argument('--flow-epochs', type=int, default=10, help="Number of epochs for Flow (default: %(default)s)")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate (default: %(default)s)")
    parser.add_argument('--seed', type=int, default=21, metavar='S', help='random seed (default: %(default)s)')
    parser.add_argument('--num-members', type=int, default=10, help='number of ensemble members (default: %(default)s)')
    parser.add_argument('--num-transformations', type=int, default=2, help='number of transformations (default: %(default)s)')
    parser.add_argument('--mask', type=str, default='half', choices=['checkerboard', 'half', 'random'], help='mask strategy for coupling layers (default: %(default)s)')
    parser.add_argument('--num_hidden', type=int, default=32, help='hidden dimension of the networks (default: %(default)s)')
    parser.add_argument('--n-layers', type=int, default=2, help='number of layers in the networks (default: %(default)s)')
    parser.add_argument('--no-discretise', action='store_false', help='do not discretise the data')
    parser.add_argument('--n-images', type=int, default=5000, help='number of images to load per class (default: %(default)s)')
    parser.add_argument('--classes', type=int, nargs='+', default=[0, 1, 2], help='classes to load (default: %(default)s, possible values 0-9)')

    args = parser.parse_args()

    LATENT_DIM = 2

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)

    # Load data
    train_loader = load_mnist(classes=args.classes, batch_size=args.batch_size, n_images=args.n_images, discretize=args.no_discretise)

    # Define VAE model
    prior = v.GaussianPrior(LATENT_DIM)

    encoder_net = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),
        nn.Softplus(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.Softplus(),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear(512, 2*LATENT_DIM),
    )

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(LATENT_DIM, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softplus(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softplus(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softplus(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    encoder = v.GaussianEncoder(encoder_net)
    if args.no_discretise:
        decoders = [v.ContinuousBernoulliDecoder(new_decoder()) for _ in range(args.num_members)]
    else:
        decoders = [v.BernoulliDecoder(new_decoder()) for _ in range(args.num_members)]
    vae = v.EVAE(prior, encoder, decoders).to(device)

    # Train models
    if args.mode in ["vae", "both"]:

        optimizer = torch.optim.AdamW(vae.parameters(), lr=args.lr)

        # Train VAE
        v.train(vae, optimizer, train_loader, args.vae_epochs, device, args.no_discretise)

        # Save VAE model
        torch.save(vae.state_dict(), f"{args.model_dir}/{args.vae_name}.pt")

    if args.mode in ["flow", "both"]:
        # load VAE model
        vae.load_state_dict(torch.load(f"{args.model_dir}/{args.vae_name}.pt"))
        vae.eval()

        # Calculate latent space
        latents = calculate_latents(vae, train_loader, device)

        latent_dataloader = torch.utils.data.DataLoader(latents, batch_size=args.batch_size, shuffle=True)

        # Define Flow model
        base = f.GaussianBase(LATENT_DIM)

        # Define transformations
        transformations =[]

        masks = f.define_masks(LATENT_DIM, args.num_transformations, strategy=args.mask)
        
        for i in range(args.num_transformations):
            scale_net = nn.Sequential(nn.Linear(LATENT_DIM, args.num_hidden), nn.ReLU(), nn.Linear(args.num_hidden, LATENT_DIM))
            translation_net = nn.Sequential(nn.Linear(LATENT_DIM, args.num_hidden), nn.ReLU(), nn.Linear(args.num_hidden, LATENT_DIM))
            transformations.append(f.MaskedCouplingLayer(scale_net, translation_net, masks[i]))

        # Define flow model
        base = f.GaussianBase(LATENT_DIM)
        flow = f.Flow(base, transformations).to(device)
        flow_optimizer = torch.optim.Adam(flow.parameters(), lr=args.lr)
        flow_train_loader = torch.utils.data.DataLoader(latents, batch_size=args.batch_size, shuffle=True)
        f.train(flow, flow_optimizer, latent_dataloader, epochs=args.flow_epochs, device=device)

        # Save Flow model
        torch.save(flow.state_dict(), f"{args.model_dir}/{args.flow_name}.pt")
    
    # Save args to file as yaml
    with open(f"{args.model_dir}/args.yaml", 'w') as f:
        yaml.dump(vars(args), f)