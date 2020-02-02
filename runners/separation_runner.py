import os
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn as nn

from models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image

__all__ = ['SeparationRunner']

BATCH_SIZE = 64
GRID_SIZE = 8

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


class SeparationRunner():
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
            # image1, image2 = get_images_split(first_digits, second_digits)
            image1, image2 = get_images_no_split(dataset)

            # mixed = (image1 + image2).float()
            mixed = torch.clamp(image1 + image2, 0, 1).float()

            mixed_grid = make_grid(mixed.detach(), nrow=GRID_SIZE, pad_value=1., padding=1)
            save_image(mixed_grid, "results/mixed_mnist.png")

            gt1_grid = make_grid(image1, nrow=GRID_SIZE, pad_value=1., padding=1)
            save_image(gt1_grid, "results/image1gt_mnist.png")

            gt2_grid = make_grid(image2, nrow=GRID_SIZE, pad_value=1., padding=1)
            save_image(gt2_grid, "results/image2gt_mnist.png")

            mixed = torch.Tensor(mixed).cuda().view(BATCH_SIZE, 1, 28, 28)

            y = nn.Parameter(torch.Tensor(BATCH_SIZE, 1, 28, 28).uniform_()).cuda()
            x = nn.Parameter(torch.Tensor(BATCH_SIZE, 1, 28, 28).uniform_()).cuda()

            step_lr=0.00002

            # Noise amounts
            sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,
                               0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
            n_steps_each = 100

            for idx, sigma in enumerate(sigmas):
                lambda_recon = 0.1/(sigma**2)
                # Not completely sure what this part is for
                labels = torch.ones(1, device=x.device) * idx
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                for step in range(n_steps_each):
                    noise_x = torch.randn_like(x) * np.sqrt(step_size * 2)
                    noise_y = torch.randn_like(y) * np.sqrt(step_size * 2)

                    grad_x = scorenet(x.view(BATCH_SIZE, 1, 28, 28), labels).detach()
                    grad_y = scorenet(y.view(BATCH_SIZE, 1, 28, 28), labels).detach()

                    #recon_loss = (torch.norm(torch.flatten(y+x-mixed)) ** 2)
                    #recon_loss = torch.norm(torch.flatten(torch.clamp(x + y, -10000000, 1) - mixed)) ** 2

                    recon_loss = torch.norm(torch.flatten((1 / (1 + torch.exp(-5 * (y + x - 0.5))))  - mixed)) ** 2
                    #recon_loss = torch.norm(torch.flatten((y - mixed)) ** 2
                    print(recon_loss)
                    recon_grads = torch.autograd.grad(recon_loss, [x,y])
                    print(recon_grads[0].mean())

                    #x = x + (step_size * grad_x) + noise_x
                    #y = y + (step_size * grad_y) + noise_y
                    x = x + (step_size * grad_x) + (-step_size * lambda_recon * recon_grads[0].detach()) + noise_x
                    y = y + (step_size * grad_y) + (-step_size * lambda_recon * recon_grads[1].detach()) + noise_y

                    # x = x + (-step_size * lambda_recon * recon_grads[0].detach()) + noise_x
                    # y = y + (-step_size * lambda_recon * recon_grads[1].detach()) + noise_y


            x = torch.clamp(x, 0, 1)
            y = torch.clamp(y, 0, 1)

            x_to_write = torch.Tensor(x.detach().cpu())
            y_to_write = torch.Tensor(y.detach().cpu())

            # PSNR Measure
            # for idx in range(BATCH_SIZE):
            #     est1 = psnr(x[idx], image1[idx].cuda()) + psnr(y[idx], image2[idx].cuda())
            #     est2 = psnr(x[idx], image2[idx].cuda()) + psnr(y[idx], image1[idx].cuda())
            #     correct_estimate = max(est1, est2) / 2.
            #     all_psnr.append(correct_estimate)

            #     if est2 > est1:
            #         x_to_write[idx] = y[idx]
            #         y_to_write[idx] = x[idx]

            # Percentage Measure
            x_thresh = (x > 0.01)
            y_thresh = (y > 0.01)
            image1_thresh = (image1 > 0.01)
            image2_thresh = (image2 > 0.01)
            avg_thresh = ((mixed.detach()[idx] / 2.) > 0.01)
            for idx in range(BATCH_SIZE):
                est1 = np.count_nonzero((x_thresh[idx] == image1_thresh[idx].cuda()).detach().cpu()) + np.count_nonzero((y_thresh[idx] == image2_thresh[idx].cuda()).detach().cpu())
                est2 = np.count_nonzero((x_thresh[idx] == image2_thresh[idx].cuda()).detach().cpu()) + np.count_nonzero((y_thresh[idx] == image1_thresh[idx].cuda()).detach().cpu())
                correct_estimate = max(est1, est2)
                percentage = correct_estimate / (2 * x.shape[-1] * x.shape[-2])
                all_percentages.append(percentage)

                dummy_count = np.count_nonzero((avg_thresh == image1_thresh[idx].cuda()).detach().cpu()) + np.count_nonzero((avg_thresh == image2_thresh[idx].cuda()).detach().cpu())
                dummy_percentage = dummy_count / (2 * x.shape[-1] * x.shape[-2])
                dummy_metrics.append(dummy_percentage)

            # Recon Grid
            recon_grid = make_grid(torch.clamp(torch.clamp(x_to_write, 0, 1) + torch.clamp(y_to_write, 0, 1), 0, 1), nrow=GRID_SIZE, pad_value=1., padding=1)
            save_image(recon_grid, "results/mnist_recon.png")
            # Write x and y
            x_grid = make_grid(x_to_write, nrow=GRID_SIZE, pad_value=1., padding=1)
            save_image(x_grid, "results/x_mnist2.png")

            y_grid = make_grid(y_to_write, nrow=GRID_SIZE, pad_value=1., padding=1)
            save_image(y_grid, "results/y_mnist2.png")

            # average_grid = make_grid(mixed.detach()/2., nrow=GRID_SIZE)
            # save_image(average_grid, "results/average_cifar.png")

            #print("Curr mean {}".format(np.array(all_psnr).mean()))
            print("Curr mean {}".format(np.array(all_percentages).mean()))
            print("Curr dummy mean {}".format(np.array(dummy_metrics).mean()))
            import pdb
            pdb.set_trace()

