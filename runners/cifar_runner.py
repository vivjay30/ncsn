import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader

__all__ = ['CifarRunner']

class CifarRunner():
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

        # Grab the first two samples from MNIST
        dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'mnist'), train=False, download=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        image0 = np.array(dataset[130][0]).astype(np.float).transpose(2, 0, 1)
        image1 = np.array(dataset[131][0]).astype(np.float).transpose(2, 0, 1)

        mixed = (image0 + image1)
        mixed = mixed / 255.

        cv2.imwrite("mixed.png", (mixed * 127.5).astype(np.uint8).transpose(1, 2, 0)[:,:,::-1])
        cv2.imwrite("gt0.png", (image0).astype(np.uint8).transpose(1, 2, 0)[:,:,::-1])
        cv2.imwrite("gt1.png", (image1).astype(np.uint8).transpose(1, 2, 0)[:,:,::-1])
        mixed = torch.Tensor(mixed).cuda()

        y = nn.Parameter(torch.Tensor(3,32,32).uniform_()).cuda()
        x = nn.Parameter(torch.Tensor(3,32,32).uniform_()).cuda()

        step_lr=0.00002

        # Noise amounts
        sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,
                           0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
        n_steps_each = 100
        lambda_recon = 1.5  # Weight to put on reconstruction error vs p(x)

        for idx, sigma in enumerate(sigmas):
            # Not completely sure what this part is for
            labels = torch.ones(1, device=x.device) * idx
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            
            for step in range(n_steps_each):
                noise_x = torch.randn_like(x) * np.sqrt(step_size * 2)
                noise_y = torch.randn_like(y) * np.sqrt(step_size * 2)

                grad_x = scorenet(x.view(1, 3, 32, 32), labels).detach()
                grad_y = scorenet(y.view(1, 3, 32, 32), labels).detach()

                recon_loss = (torch.norm(torch.flatten(y+x-mixed)) ** 2)
                print(recon_loss)
                recon_grads = torch.autograd.grad(recon_loss, [x,y])

                #x = x + (step_size * grad_x) + noise_x
                #y = y + (step_size * grad_y) + noise_y
                x = x + (step_size * grad_x) + (-step_size * lambda_recon * recon_grads[0].detach()) + noise_x
                y = y + (step_size * grad_y) + (-step_size * lambda_recon * recon_grads[1].detach()) + noise_y

            lambda_recon *= 2.8

        # Write x and y
        x_np = x.detach().cpu().numpy()[0,:,:,:]
        x_np = np.clip(x_np, 0, 1)
        cv2.imwrite("x.png", (x_np * 255).astype(np.uint8).transpose(1, 2, 0)[:,:,::-1])

        y_np = y.detach().cpu().numpy()[0,:,:,:]
        y_np = np.clip(y_np, 0, 1)
        cv2.imwrite("y.png", (y_np * 255).astype(np.uint8).transpose(1, 2, 0)[:,:,::-1])

        cv2.imwrite("out_mixed.png", (y_np * 127.5).astype(np.uint8).transpose(1, 2, 0)[:,:,::-1] + (x_np * 127.5).astype(np.uint8).transpose(1, 2, 0)[:,:,::-1])

        import pdb
        pdb.set_trace()

