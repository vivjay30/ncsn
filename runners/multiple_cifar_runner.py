import os
from copy import deepcopy
from itertools import permutations

import cv2
import numpy as np
import torch
import torch.nn as nn

from models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image

__all__ = ['MultipleCIFARRunner']

BATCH_SIZE = 100
GRID_SIZE = 10
N = 2  # Number of digits
SAVE_DIR = "results/output_dirs/cifar_10x10_{}".format(N)

def psnr(est, gt):
    """Returns the P signal to noise ratio between the estimate and gt"""
    return float(-10 * torch.log10(((est - gt) ** 2).mean()).detach().cpu())

def gehalf(input_tensor): 
    """Returns a sigmoid proxy for x > 0.5"""
    return 1 / (1 + torch.exp(-5 * (input_tensor - 0.5)))

def get_single_image(data):
    """Returns two images, one from [0,4] and the other from [5,9]"""
    rand_idx = np.random.randint(0, data.shape[0] - 1, BATCH_SIZE)
    image = torch.Tensor(data[rand_idx]).float().view(BATCH_SIZE, 3, 32, 32) / 255.

    return image


class MultipleCIFARRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def test(self):
        all_psnr = []  # All signal to noise ratios over all the batches
        all_percentages = []  # All percentage accuracies
        dummy_metrics = []  # Metrics for the averaging value

        # Load the score network
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet)
        scorenet.load_state_dict(states[0])
        scorenet.eval()

        # Grab the first two samples from MNIST
        trans = transforms.Compose([transforms.ToTensor()])
        dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=False, download=True)
        data = dataset.data.transpose(0, 3, 1, 2)


        for iteration in range(100):
            print("Iteration {}".format(iteration))
            curr_dir = os.path.join(SAVE_DIR, "{:07d}".format(iteration))
            if not os.path.exists(curr_dir):
                os.makedirs(curr_dir)
            # image1, image2 = get_images_split(first_digits, second_digits)
            gt_images = []
            for _ in range(N):
                gt_images.append(get_single_image(data))

            mixed = sum(gt_images).float()

            mixed_grid = make_grid(mixed.detach() / float(N), nrow=GRID_SIZE, pad_value=1., padding=1)
            save_image(mixed_grid, os.path.join(curr_dir, "mixed.png"))

            for i in range(N):
                gt_grid = make_grid(gt_images[i], nrow=GRID_SIZE, pad_value=1., padding=1)
                save_image(gt_grid, os.path.join(curr_dir, "gt{}.png".format(i)))

            mixed = torch.Tensor(mixed).cuda().view(BATCH_SIZE, 3, 32, 32)

            xs = []
            for _ in range(N):
                xs.append(nn.Parameter(torch.Tensor(BATCH_SIZE, 3, 32, 32).uniform_()).cuda())

            step_lr=0.00002

            # Noise amounts
            sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,
                              0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
            n_steps_each = 200

            for idx, sigma in enumerate(sigmas):
                lambda_recon = 1.8/(sigma**2)
                # Not completely sure what this part is for
                labels = torch.ones(1, device=xs[0].device) * idx
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                for step in range(n_steps_each):
                    noises = []
                    for _ in range(N):
                        noises.append(torch.randn_like(xs[0]) * np.sqrt(step_size * 2))

                    grads = []
                    for i in range(N):
                        grads.append(scorenet(xs[i].view(BATCH_SIZE, 3, 32, 32), labels).detach())

                    recon_loss = (torch.norm(torch.flatten(sum(xs) - mixed)) ** 2)
                    print(recon_loss)
                    recon_grads = torch.autograd.grad(recon_loss, xs)

                    for i in range(N):
                        xs[i] = xs[i] + (step_size * grads[i]) + (-step_size * lambda_recon * recon_grads[i].detach()) + noises[i]

            for i in range(N):
                xs[i] = torch.clamp(xs[i], 0, 1)

            x_to_write = []
            for i in range(N):
                x_to_write.append(torch.Tensor(xs[i].detach().cpu()))

            # PSNR Measure
            for idx in range(BATCH_SIZE):
                best_psnr = -10000
                best_permutation = None
                for permutation in permutations(range(N)):
                    curr_psnr = sum([psnr(xs[permutation[i]][idx], gt_images[i][idx].cuda()) for i in range(N)])
                    if curr_psnr > best_psnr:
                        best_psnr = curr_psnr
                        best_permutation = permutation

                all_psnr.append(best_psnr / float(N))
                for i in range(N):
                    x_to_write[i][idx] = xs[best_permutation[i]][idx] 

                    mixed_psnr = psnr(mixed.detach().cpu()[idx] / float(N), gt_images[i][idx])
                    dummy_metrics.append(mixed_psnr)
                
            for i in range(N):
                x_grid = make_grid(x_to_write[i], nrow=GRID_SIZE, pad_value=1., padding=1)
                save_image(x_grid, os.path.join(curr_dir, "x{}.png".format(i)))

            mixed_grid = make_grid(sum(xs)/float(N), nrow=GRID_SIZE, pad_value=1., padding=1)
            save_image(mixed_grid, os.path.join(curr_dir, "recon.png".format(i)))


            # average_grid = make_grid(mixed.detach()/2., nrow=GRID_SIZE)
            # save_image(average_grid, "results/average_cifar.png")
            
            print("Curr mean {}".format(np.array(all_psnr).mean()))
            print("Const mean {}".format(np.array(dummy_metrics).mean()))

