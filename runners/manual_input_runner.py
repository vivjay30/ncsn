import os
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn as nn

from models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image


__all__ = ['ManualInputRunner']

BATCH_SIZE = 29
GRID_SIZE = 6
SAVE_DIR = "results/output_dirs/cifar_manual_inputs/"

def psnr(est, gt):
    """Returns the P signal to noise ratio between the estimate and gt"""
    return float(-10 * torch.log10(((est - gt) ** 2).mean()).detach().cpu())

def process_grid(grid, image_dim, border_dim, grid_size):
    output_images = []
    for h_idx in range(grid_size[0]):
        for w_idx in range(grid_size[1]):
            start_h = h_idx * image_dim + ((h_idx + 1) * border_dim)
            end_h = (h_idx + 1) * (image_dim + border_dim)

            start_w = w_idx * image_dim + ((w_idx + 1) * border_dim)
            end_w = (w_idx + 1) * (image_dim + border_dim)

            image = grid[start_h:end_h, start_w:end_w, :]

            # Check for black images
            if image.mean() > 0:
                output_images.append(grid[start_h:end_h, start_w:end_w, :])
    return output_images


def get_images_manual():
    gt1 = []
    gt2 = []

    inputs = [
        ["inputs/gt1_1.png", (3, 8)],
        ["inputs/gt1_2.png", (3, 8)],
        ["inputs/gt2_1.png", (2, 8)],
        ["inputs/gt2_2.png", (2, 8)],
    ]

    for idx in range(len(inputs)):
        grid_image = cv2.imread(inputs[idx][0])[:,:,::-1]
        if idx in [0, 2]:
            gt1 += process_grid(grid_image, 32, 2, inputs[idx][1])
        else:
            gt2 += process_grid(grid_image, 32, 2, inputs[idx][1])

    return torch.Tensor(np.array(gt1).transpose(0, 3, 1, 2)) / 255., torch.Tensor(np.array(gt2).transpose(0, 3, 1, 2)) / 255.

class ManualInputRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def test(self):
        # Load the score network
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet)
        scorenet.load_state_dict(states[0])
        scorenet.eval()


        image1, image2 = get_images_manual()
        mixed = (image1 + image2).float()
        curr_dir = SAVE_DIR

        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)

        mixed_grid = make_grid(mixed.detach() / 2., nrow=GRID_SIZE)
        save_image(mixed_grid, os.path.join(curr_dir, "mixed.png"))

        gt1_grid = make_grid(image1, nrow=GRID_SIZE)
        save_image(gt1_grid, os.path.join(curr_dir, "gt1.png"))

        gt2_grid = make_grid(image2, nrow=GRID_SIZE)
        save_image(gt2_grid, os.path.join(curr_dir, "gt2.png"))

        mixed = torch.Tensor(mixed).cuda().view(BATCH_SIZE, 3, 32, 32)

        y = nn.Parameter(torch.Tensor(BATCH_SIZE, 3, 32, 32).uniform_()).cuda()
        x = nn.Parameter(torch.Tensor(BATCH_SIZE, 3, 32, 32).uniform_()).cuda()

        step_lr=0.00002

        # Noise amounts
        sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,
                           0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
        n_steps_each = 100

        #lambda_recon = 1.5
        for idx, sigma in enumerate(sigmas):
            lambda_recon  = 1.8 / (sigma ** 2)

            # Not completely sure what this part is for
            labels = torch.ones(1, device=x.device) * idx
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            for step in range(n_steps_each):
                noise_x = torch.randn_like(x) * np.sqrt(step_size * 2)
                noise_y = torch.randn_like(y) * np.sqrt(step_size * 2)

                grad_x = scorenet(x.view(BATCH_SIZE, 3, 32, 32), labels).detach()
                grad_y = scorenet(y.view(BATCH_SIZE, 3, 32, 32), labels).detach()

                recon_loss = (torch.norm(torch.flatten(y+x-mixed)) ** 2)
                print(recon_loss)
                recon_grads = torch.autograd.grad(recon_loss, [x,y])

                x = x + (step_size * grad_x) + (-step_size * lambda_recon * recon_grads[0].detach()) + noise_x
                y = y + (step_size * grad_y) + (-step_size * lambda_recon * recon_grads[1].detach()) + noise_y


        x_to_write = torch.Tensor(x.detach().cpu())
        y_to_write = torch.Tensor(y.detach().cpu())


        for idx in range(BATCH_SIZE):
            # PSNR
            est1 = psnr(x[idx], image1[idx].cuda()) + psnr(y[idx], image2[idx].cuda())
            est2 = psnr(x[idx], image2[idx].cuda()) + psnr(y[idx], image1[idx].cuda())

            if est2 > est1:
                x_to_write[idx] = y[idx]
                y_to_write[idx] = x[idx]

        # # Write x and y
        x_grid = make_grid(x_to_write, nrow=GRID_SIZE)
        save_image(x_grid, os.path.join(curr_dir, "x.png"))

        y_grid = make_grid(y_to_write, nrow=GRID_SIZE)
        save_image(y_grid, os.path.join(curr_dir, "y.png"))






