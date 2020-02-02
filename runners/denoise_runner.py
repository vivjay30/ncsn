import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from datasets.celeba import CelebA

from models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader

__all__ = ['DenoiseRunner']

class DenoiseRunner():
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
        dataset = CelebA(os.path.join(self.args.run, 'datasets', 'celeba'), split='test', download=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        input_image = np.array(dataset[0][0]).astype(np.float).transpose(2, 0, 1)

        # input_image = cv2.imread("/projects/grail/vjayaram/source_separation/ncsn/run/datasets/celeba/celeba/img_align_celeba/012690.jpg")

        # input_image = cv2.resize(input_image, (32, 32))[:,:,::-1].transpose(2, 0, 1)
        input_image = input_image / 255.
        noise = np.random.randn(*input_image.shape) / 10
        cv2.imwrite("input_image.png", (input_image * 255).astype(np.uint8).transpose(1, 2, 0)[:,:,::-1])
        input_image += noise
        input_image = np.clip(input_image, 0, 1)

        cv2.imwrite("input_image_noisy.png", (input_image * 255).astype(np.uint8).transpose(1, 2, 0)[:,:,::-1])

        input_image = torch.Tensor(input_image).cuda()
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

                grad_x = scorenet(x.view(1, 3, 32, 32), labels).detach()

                recon_loss = (torch.norm(torch.flatten(input_image - x)) ** 2)
                print(recon_loss)
                recon_grads = torch.autograd.grad(recon_loss, [x])

                #x = x + (step_size * grad_x) + noise_x
                x = x + (step_size * grad_x) + (-step_size * lambda_recon * recon_grads[0].detach()) + noise_x

            lambda_recon *= 1.6

        # # Write x and y
        x_np = x.detach().cpu().numpy()[0,:,:,:]
        x_np = np.clip(x_np, 0, 1)
        cv2.imwrite("x.png", (x_np * 255).astype(np.uint8).transpose(1, 2, 0)[:,:,::-1])

        # y_np = y.detach().cpu().numpy()[0,:,:,:]
        # y_np = np.clip(y_np, 0, 1)
        # cv2.imwrite("y.png", (y_np * 255).astype(np.uint8).transpose(1, 2, 0)[:,:,::-1])

        # cv2.imwrite("out_mixed.png", (y_np * 127.5).astype(np.uint8).transpose(1, 2, 0)[:,:,::-1] + (x_np * 127.5).astype(np.uint8).transpose(1, 2, 0)[:,:,::-1])

        import pdb
        pdb.set_trace()

