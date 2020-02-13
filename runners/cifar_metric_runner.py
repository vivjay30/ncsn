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

from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as ssim

import sys
sys.path.append("/projects/grail/vjayaram/source_separation/improved-gan/inception_score/")

# def get_ssim(est, act):
#     ssim = 0
#     for i in range(len(est)):
#         ssim += compare_ssim(np.transpose(est[i], (1, 2, 0)),
#                              np.transpose(act[i], (1, 2, 0)),
#                              data_range=1, multichannel=True)
#     return ssim / len(est)

ANIMAL_IDX = [2, 3, 4, 5, 6, 7]
MACHINE_IDX = [0, 1, 8, 9]


def get_images_split(first_items, second_items):
    """Returns two images, one from the datset first_items and the other from
       second_items. Both should be numpy arryas in N x 3 x 32 x 32 shape"""
    rand_idx_1 = np.random.randint(0, first_items.shape[0] - 1, BATCH_SIZE)
    rand_idx_2 = np.random.randint(0, second_items.shape[0] - 1, BATCH_SIZE)

    image1 = torch.Tensor(first_items[rand_idx_1]).float().view(BATCH_SIZE, 3, 32, 32) / 255.
    image2 = torch.Tensor(second_items[rand_idx_2]).float().view(BATCH_SIZE, 3, 32, 32) / 255.

    return image1, image2

def get_ssim(est, act):
    return ssim(est.detach().cpu().numpy().transpose(1,2,0),
                act.numpy().transpose(1,2,0),
                data_range=1.0,
                multichannel=True)


def get_ssim_grayscale(est, act):
    return ssim(est.mean(0).detach().cpu().numpy(),
                act.mean(0).numpy(),
                data_range=1.0,
                multichannel=False)

__all__ = ['CIFARMetricRunner']

BATCH_SIZE = 100
GRID_SIZE = 10
SAVE_DIR = "results/output_dirs/cifar_inception_class_agnostic/"

def psnr(est, gt):
    """Returns the P signal to noise ratio between the estimate and gt"""
    return float(-10 * torch.log10(((est - gt) ** 2).mean()).detach().cpu())

class CIFARMetricRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def test(self):
        # For metrics
        all_psnr = []  # All signal to noise ratios over all the batches
        all_grayscale_psnr = []  # All signal to noise ratios over all the batches

        all_mixed_psnr = []  # All signal to noise ratios over all the batches
        all_mixed_grayscale_psnr = []  # All signal to noise ratios over all the batches

        all_ssim = []
        all_grayscale_ssim = []

        all_mixed_ssim = []  # All signal to noise ratios over all the batches
        all_mixed_grayscale_ssim = []  # All signal to noise ratios over all the batches
        
        strange_cases = {"gt1":[], "gt2":[], "mixed":[], "x":[], "y":[]}

        # For inception score
        output_to_incept = []
        mixed_to_incept = []

        # For videos
        all_x = []
        all_y = []
        all_mixed = []

        # Load the score network
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet)
        scorenet.load_state_dict(states[0])
        scorenet.eval()

        # Grab the first two samples from MNIST
        dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=False, download=True)
        data = dataset.data.transpose(0, 3, 1, 2)
        all_animals_idx = np.isin(dataset.targets, ANIMAL_IDX)
        all_machines_idx = np.isin(dataset.targets, MACHINE_IDX)

        all_animals = dataset.data[all_animals_idx].transpose(0, 3, 1, 2)
        all_machines = dataset.data[all_machines_idx].transpose(0, 3, 1, 2)

        for iteration in range(105, 250):
            print("Iteration {}".format(iteration))
            curr_dir = os.path.join(SAVE_DIR, "{:07d}".format(iteration))
            if not os.path.exists(curr_dir):
                os.makedirs(curr_dir)

            # rand_idx_1 = np.random.randint(0, data.shape[0] - 1, BATCH_SIZE)
            # rand_idx_2 = np.random.randint(0, data.shape[0] - 1, BATCH_SIZE)

            # image1 = torch.tensor(data[rand_idx_1, :].astype(np.float) / 255.).float()
            # image2 = torch.tensor(data[rand_idx_2, :].astype(np.float) / 255.).float()

            image1, image2 = get_images_split(all_animals, all_machines)
            mixed = (image1 + image2).float()
            
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
                lambda_recon  = 1.0 / (sigma ** 2)

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
                    recon_grads = torch.autograd.grad(recon_loss, [x,y])

                    #x = x + (step_size * grad_x) + noise_x
                    #y = y + (step_size * grad_y) + noise_y
                    x = x + (step_size * grad_x) + (-step_size * lambda_recon * recon_grads[0].detach()) + noise_x
                    y = y + (step_size * grad_y) + (-step_size * lambda_recon * recon_grads[1].detach()) + noise_y

                    # Video
                    # if (step % 5) == 0:
                    #     all_x.append(x.detach().cpu().numpy())
                    #     all_y.append(y.detach().cpu().numpy())
                    #     all_mixed.append((x.detach().cpu().numpy() + y.detach().cpu().numpy()) / 2.)
                #lambda_recon *= 2.8

            # Inception
            # from model import get_inception_score
            # output_to_incept += [np.clip(x[idx].detach().cpu().numpy().transpose(1,2,0), 0, 1) * 255. for idx in range(x.shape[0])]
            # output_to_incept += [np.clip(y[idx].detach().cpu().numpy().transpose(1,2,0), 0, 1) * 255. for idx in range(y.shape[0])]
            # mixed_to_incept += [np.clip((mixed[idx] / 2.).detach().cpu().numpy().transpose(1,2,0), 0, 1) * 255. for idx in range(y.shape[0])]
            # mixed_to_incept += [np.clip((mixed[idx] / 2.).detach().cpu().numpy().transpose(1,2,0), 0, 1) * 255. for idx in range(y.shape[0])]
            

            x_to_write = torch.Tensor(x.detach().cpu())
            y_to_write = torch.Tensor(y.detach().cpu())

            # x_movie = np.array(np.stack(all_x, axis=0))
            # y_movie = np.array(np.stack(all_y, axis=0))
            # mixed_movie = np.array(np.stack(all_mixed, axis=0))


            for idx in range(BATCH_SIZE):
                # PSNR
                est1 = psnr(x[idx], image1[idx].cuda()) + psnr(y[idx], image2[idx].cuda())
                est2 = psnr(x[idx], image2[idx].cuda()) + psnr(y[idx], image1[idx].cuda())
                correct_psnr = max(est1, est2) / 2.
                all_psnr.append(correct_psnr)

                grayscale_est1 = psnr(x[idx].mean(0), image1[idx].mean(0).cuda()) + psnr(y[idx].mean(0), image2[idx].mean(0).cuda())
                grayscale_est2 = psnr(x[idx].mean(0), image2[idx].mean(0).cuda()) + psnr(y[idx].mean(0), image1[idx].mean(0).cuda())
                grayscale_psnr = max(grayscale_est1, grayscale_est2) / 2.
                all_grayscale_psnr.append(grayscale_psnr)

                # Mixed PSNR
                mixed_psnr = psnr((mixed[idx] / 2.), image1[idx].cuda())
                all_mixed_psnr.append(mixed_psnr)

                grayscale_mixed_psnr = psnr((mixed[idx] / 2.).mean(0), image1[idx].mean(0).cuda())
                all_mixed_grayscale_psnr.append(grayscale_mixed_psnr)

                if est2 > est1:
                    x_to_write[idx] = y[idx]
                    y_to_write[idx] = x[idx]

                    # tmp = x_movie[:, idx].copy()
                    # x_movie[:, idx] = y_movie[:, idx]
                    # y_movie[:, idx] = tmp

                # SSIM
                est1 = get_ssim(x[idx], image1[idx]) + get_ssim(y[idx], image2[idx])
                est2 = get_ssim(x[idx], image2[idx]) + get_ssim(y[idx], image1[idx])
                correct_ssim = max(est1, est2) / 2.
                all_ssim.append(correct_ssim)

                grayscale_est1 = get_ssim_grayscale(x[idx], image1[idx]) + get_ssim(y[idx], image2[idx])
                grayscale_est2 = get_ssim_grayscale(x[idx], image2[idx]) + get_ssim(y[idx], image1[idx])
                grayscale_ssim = max(grayscale_est1, grayscale_est2) / 2.
                all_grayscale_ssim.append(grayscale_ssim)

                # Mixed ssim
                mixed_ssim = get_ssim((mixed[idx] / 2.), image1[idx]) + get_ssim((mixed[idx] / 2.), image2[idx])
                all_mixed_ssim.append(mixed_ssim / 2.)

                grayscale_mixed_ssim = get_ssim_grayscale((mixed[idx] / 2.), image1[idx]) + get_ssim_grayscale((mixed[idx] / 2.), image2[idx])
                all_mixed_grayscale_ssim.append(grayscale_mixed_ssim / 2.)

                if correct_psnr < 19 and grayscale_psnr > 20.5:
                    strange_cases["gt1"].append(image1[idx].detach().cpu().numpy())
                    strange_cases["gt2"].append(image2[idx].detach().cpu().numpy())
                    strange_cases["mixed"].append(mixed[idx].detach().cpu().numpy())
                    strange_cases["x"].append(x_to_write[idx].detach().cpu().numpy())
                    strange_cases["y"].append(y_to_write[idx].detach().cpu().numpy())
                    print("Added strange case")


            # # Write x and y
            x_grid = make_grid(x_to_write, nrow=GRID_SIZE)
            save_image(x_grid, os.path.join(curr_dir, "x.png"))

            y_grid = make_grid(y_to_write, nrow=GRID_SIZE)
            save_image(y_grid, os.path.join(curr_dir, "y.png"))

            print("PSNR {}".format(np.array(all_psnr).mean()))
            print("Mixed PSNR {}".format(np.array(all_mixed_psnr).mean()))

            print("PSNR Grayscale {}".format(np.array(all_grayscale_psnr).mean()))
            print("Mixed PSNR Grayscale {}".format(np.array(all_mixed_grayscale_psnr).mean()))

            print("SSIM {}".format(np.array(all_ssim).mean()))
            print("Mixed SSIM {}".format(np.array(all_mixed_ssim).mean()))

            print("SSIM Grayscale {}".format(np.array(all_grayscale_ssim).mean()))            
            print("Mixed SSIM Grayscale {}".format(np.array(all_mixed_grayscale_ssim).mean()))


            # Write video frames
            # padding = 50
            # dim_w = 172 * 4 + padding * 5
            # dim_h = 172 * 2 + padding * 3
            # for frame_idx in range(x_movie.shape[0]):
            #     print(frame_idx)
            #     x_grid = make_grid(torch.Tensor(x_movie[frame_idx]), nrow=GRID_SIZE)
            #     # save_image(x_grid, "results/videos/x/x_{}.png".format(frame_idx))

            #     y_grid = make_grid(torch.Tensor(y_movie[frame_idx]), nrow=GRID_SIZE)
            #     # save_image(y_grid, "results/videos/y/y_{}.png".format(frame_idx))

            #     recon_grid = make_grid(torch.Tensor(mixed_movie[frame_idx]), nrow=GRID_SIZE)
            #     # save_image(recon_grid, "results/videos/mixed/mixed_{}.png".format(frame_idx))


            #     output_frame = torch.zeros(3, dim_h, dim_w)
            #     output_frame[:, 50:(50+172), 50:(50+172)] = gt1_grid
            #     output_frame[:, (100+172):(100+172*2), 50:(50+172)] = gt2_grid
            #     output_frame[:, (75 + 86):(75 + 86 + 172), (50 * 2 + 172):(50 * 2 + 172 * 2)] = mixed_grid
            #     output_frame[:, (75 + 86):(75 + 86 + 172), (50 * 3 + 172 * 2):(50 * 3 + 172 * 3)] = recon_grid
            #     output_frame[:,50:(50 + 172),(50 * 4 + 172 * 3):(50 * 4 + 172 * 4)] = x_grid
            #     output_frame[:,(50 * 2 + 172):(50 * 2 + 172 * 2),(50 * 4 + 172 * 3):(50 * 4 + 172 * 4)] = y_grid
            #     save_image(output_frame, "results/videos/combined/{:03d}.png".format(frame_idx))


        # Calculate inception scores
        # print("Output inception score {}".format(get_inception_score(output_to_incept)))
        # print("Mixed inception score {}".format(get_inception_score(mixed_to_incept)))

        # Write strange results
        y1 = np.stack(strange_cases["y"], axis=0)
        y_grid = make_grid(torch.Tensor(y1))
        save_image(y_grid, "results/y_strange.png")

        x1 = np.stack(strange_cases["x"], axis=0)
        x_grid = make_grid(torch.Tensor(x1))
        save_image(x_grid, "results/x_strange.png")

        gt1 = np.stack(strange_cases["gt1"], axis=0)
        gt1_grid = make_grid(torch.Tensor(gt1))
        save_image(gt1_grid, "results/gt1_strange.png")

        gt2 = np.stack(strange_cases["gt2"], axis=0)
        gt2_grid = make_grid(torch.Tensor(gt2))
        save_image(gt2_grid, "results/gt2_strange.png")

        mixed = np.stack(strange_cases["mixed"], axis=0) / 2.
        mixed_grid = make_grid(torch.Tensor(mixed))
        save_image(mixed_grid, "results/mixed_strange.png")





