import models.vae as v
import models.flow as f
from geodesics import optimize_curve
from utils import calculate_latents
from data.mnist import load_mnist

import customtkinter as ctk
import torch
import matplotlib.pyplot as plt
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from argparse import ArgumentParser


class GUI(ctk.CTk):
    figure = None
    point = None
    curve = None

    def __init__(self, model, data_loader, device, N=50):
        super().__init__()
        self.N = N
        self.model = model
        self.device = device

        self.title("Geodesics Illustration")
        self.geometry("1280x720")

        # Create sidebar
        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.pack(side="left", fill="y")

        # Create main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(side="right", fill="both", expand=True)

        # Create plt figure
        self.canvas = self.initialise_plot(self.model, data_loader)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.canvas.draw()

        # Add red cross to canvas on click
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Esqape to close
        self.bind("<Escape>", lambda e: self.quit())
    
    def on_click(self, event):
        if event.inaxes is None:
            return
        if self.point is None:
            self.point = self.canvas.figure.axes[0].scatter(event.xdata, event.ydata, color='r', marker="x")
        else:
            # Remove old point and draw line between points
            old_point_x, old_point_y = self.point.get_offsets()[0]
            self.point.remove()
            self.point = None
            self.optimize_latent_curve((old_point_x, old_point_y), (event.xdata, event.ydata))
        self.canvas.draw()
    
    def optimize_latent_curve(self, z0, z1):
        z0 = torch.tensor(z0).float().to(self.device)
        z1 = torch.tensor(z1).float().to(self.device)
        points = torch.linspace(0, 1, self.N)
        # Interpolate between the two tensors
        interpolated_points = (z0 * (1 - points[:, None]) + z1 * points[:, None])[1:-1]

        interpolated_points.requires_grad = True
        optimizer = torch.optim.AdamW([interpolated_points], lr=0.01)

        all_points = torch.cat([z0[None], interpolated_points, z1[None]], dim=0)

        self.plot_curve(all_points)

        curve_points = optimize_curve_gui(self.plot_curve, z0, z1, interpolated_points, optimizer, max_iter=2000, min_iter=100, threshold=50, num_members=args.num_members, num_samples=args.num_members)
        
        all_points = torch.cat([z0[None], curve_points, z1[None]], dim=0)
        self.plot_curve(all_points, keep=True)
    
    def plot_curve(self, all_points, keep=False):
        all_points = all_points.detach().cpu()
        if self.curve is not None:
            for line in self.curve:
                line.remove()
        if keep:
            self.curve = None
            self.canvas.figure.axes[0].plot(all_points[:, 0], all_points[:, 1], 'g')
        else:
            self.curve = self.canvas.figure.axes[0].plot(all_points[:, 0], all_points[:, 1], 'r')
        self.canvas.draw()
        self.update()

    
    def initialise_plot(self, model, dataloader):
        ## Encode test and train data
        latents, labels = [], []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                z = model.encoder(x)
                latents.append(z.mean)
                labels.append(y)
            latents = torch.concatenate(latents, dim=0)
            labels = torch.concatenate(labels, dim=0)

        # meshgrid of latent space
        x = torch.linspace(-10, 10, 100)
        y = torch.linspace(-10, 10, 100)
        xx, yy = torch.meshgrid(x, y)
        latent_grid = torch.stack([xx, yy], dim=-1).to(self.device)
        latent_grid = latent_grid.view(-1, 2)

        ## Plot entropy of decoder
        self.fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        with torch.no_grad():
            # Compute the variance in means of the decoders
            decoder_means = torch.stack([decoder(latent_grid).mean for decoder in model.decoders], dim=0)
            decoder_variances = decoder_means.var(dim=0).mean(dim=(1, 2, 3)).view(100, 100)
        ax.pcolormesh(xx.cpu().numpy(), yy.cpu().numpy(), decoder_variances, cmap='viridis')
        # Add colorbar to fig
        plt.tight_layout()

        class_colors = ["#4ec1f3", "#dafc24", "#2405e5"]
        ## Plot training data
        for k in range(num_classes):
            idx = labels == k
            ax.scatter(latents[idx, 0], latents[idx, 1], s=1, c=class_colors[k])
        
        return FigureCanvasTkAgg(self.fig, self.main_frame)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="models/vae_mnist")

    args = parser.parse_args()

    with open(f"{args.model_dir}/args.yml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Load models based on config and rewrite GUI