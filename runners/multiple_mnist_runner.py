import os
from copy import deepcopy
from itertools import permutations

import cv2
import numpy as np
import torch
import torch.nn as nn

from models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image

__all__ = ['MultipleMNISTRunner']

BATCH_SIZE = 64
GRID_SIZE = 8
N = 6  # Number of digits
SAVE_DIR = "results/output_dirs/mnist_multiple_{}".format(N)

def psnr(est, gt):
    """Returns the P signal to noise ratio between the estimate and gt"""
    return float(-10 * torch.log10(((est - gt) ** 2).mean()).detach().cpu())

def gehalf(input_tensor):
    """Returns a sigmoid proxy for x > 0.5"""
    return 1 / (1 + torch.exp(-5 * (input_tensor - 0.5)))

def get_images_split(first_digits, second_digits):
    """Returns two images, one from [0,4] and the other from [5,9]"""
    rand_idx_1 = np.random.randint(0, first_digits.shape[0] - 1, BATCH_SIZE)
    rand_idx_2 = np.random.randint(0, second_digits.shape[0] - 1, BATCH_SIZE)

    image1 = first_digits[rand_idx_1, :, :].float().view(BATCH_SIZE, 1, 28, 28) / 255.
    image2 = second_digits[rand_idx_2, :, :].float().view(BATCH_SIZE, 1, 28, 28) / 255.

    return image1, image2

def get_images_no_split(dataset):
    image1_batch = torch.zeros(BATCH_SIZE, 28, 28)
    image2_batch = torch.zeros(BATCH_SIZE, 28, 28)
    for idx in range(BATCH_SIZE):
        idx1 = np.random.randint(0, len(dataset))
        image1 = dataset.data[idx1]
        image1_label = dataset[idx1][1]
        image2_label = image1_label

        # Continously sample image2 until not same label
        while image1_label == image2_label:
            idx2 = np.random.randint(0, len(dataset))
            image2 = dataset.data[idx2]
            image2_label = dataset[idx2][1]

        image1_batch[idx] = image1
        image2_batch[idx] = image2

    image1_batch = image1_batch.float().view(BATCH_SIZE, 1, 28, 28) / 255.
    image2_batch = image2_batch.float().view(BATCH_SIZE, 1, 28, 28) / 255.

    return image1_batch, image2_batch

def get_single_image(dataset):
    """Returns two images, one from [0,4] and the other from [5,9]"""
    rand_idx = np.random.randint(0, dataset.data.shape[0] - 1, BATCH_SIZE)
    image = dataset.data[rand_idx].float().view(BATCH_SIZE, 1, 28, 28) / 255.

    return image


class MultipleMNISTRunner():
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
        dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=False, download=True)

        first_digits_idx = dataset.train_labels <= 4
        second_digits_idx = dataset.train_labels >=5

        first_digits = dataset.train_data[first_digits_idx]
        second_digits = dataset.train_data[second_digits_idx]



        for iteration in range(100):
            print("Iteration {}".format(iteration))
            curr_dir = os.path.join(SAVE_DIR, "{:07d}".format(iteration))
            if not os.path.exists(curr_dir):
                os.makedirs(curr_dir)
            # image1, image2 = get_images_split(first_digits, second_digits)
            gt_images = []
            for _ in range(N):
                gt_images.append(get_single_image(dataset))

            mixed = sum(gt_images).float()

            mixed_grid = make_grid(mixed.detach() / float(N), nrow=GRID_SIZE, pad_value=1., padding=1)
            save_image(mixed_grid, os.path.join(curr_dir, "mixed.png"))

            for i in range(N):
                gt_grid = make_grid(gt_images[i], nrow=GRID_SIZE, pad_value=1., padding=1)
                save_image(gt_grid, os.path.join(curr_dir, "gt{}.png".format(i)))

            mixed = torch.Tensor(mixed).cuda().view(BATCH_SIZE, 1, 28, 28)

            xs = []
            for _ in range(N):
                xs.append(nn.Parameter(torch.Tensor(BATCH_SIZE, 1, 28, 28).uniform_()).cuda())

            step_lr=0.00002

            # Noise amounts
            sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,
                               0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
            n_steps_each = 100

            for idx, sigma in enumerate(sigmas):
                # if idx <= 5:
                #     lambda_recon = 1./(sigma**2)
                # else:
                #     lambda_recon = 2./(sigma**2)
                lambda_recon = 0.5/(sigma**2)
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
                        grads.append(scorenet(xs[i].view(BATCH_SIZE, 1, 28, 28), labels).detach())

                    recon_loss = (torch.norm(torch.flatten(sum(xs) - mixed)) ** 2)
                    recon_grads = torch.autograd.grad(recon_loss, xs)

                    for i in range(N):
                        xs[i] = xs[i] + (step_size * grads[i]) + (-step_size * lambda_recon * recon_grads[i].detach()) + noises[i]

                    # x = x + (-step_size * lambda_recon * recon_grads[0].detach()) + noise_x
                    # y = y + (-step_size * lambda_recon * recon_grads[1].detach()) + noise_y

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

