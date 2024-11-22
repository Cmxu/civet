import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch.nn as nn
import math
import copy
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import utils
from tqdm import tqdm
from torch.autograd import Variable


def train_model(model, 
                model_name,
                dataset, 
                epochs=20,
                batch_size=32, 
                lr=1e-04, 
                weight_decay=1e-5,
                checkpoint_dir='./models',
                device='cuda',
                training_type = 'standard',
                epsilon = 0.1,
                pgd_step_size = 0.001,
                pgd_iterations = 20,
                deltas = [0.05],
                civet_standard_iters = 100,
                civet_rampup_iters = 100,
                max_bs_depth = 20,
                loss_scaling = 1,
                tau = 0.1,
):
    deltas.sort(reverse = True)
    model.train()
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=weight_decay,
    )
    if model.ibp:
        model.unrobust()
    running_eps = 0
    itr = 0
    for epoch in range(epochs):
        data_loader = utils.get_dataloader(dataset, batch_size, device=device)['train']
        data_stream = tqdm(enumerate(data_loader))

        for batch_index, (x, _) in data_stream:
            optimizer.zero_grad()

            x = Variable(x)
            (mean, logvar), x_reconstructed = model(x)
            reconstruction_loss = model.reconstruction_loss(x_reconstructed, x)
            kl_divergence_loss = model.kl_divergence_loss(mean, logvar)
            total_loss = reconstruction_loss + kl_divergence_loss

            if training_type == 'pgd':
                v = utils.vae_pgd(model, x, device = device, step_size = pgd_step_size, epsilon = epsilon, iterations = pgd_iterations)
                (meanv, logvarv), x_rv = model(x + v)
                reconstruction_lossv = model.reconstruction_loss(x_rv, x)
                kl_divergence_lossv = model.kl_divergence_loss(meanv, logvarv)
                total_loss += loss_scaling * (reconstruction_lossv + kl_divergence_lossv)
            elif training_type == 'civet':
                if itr < civet_standard_iters:
                    cur_eps = 0
                if itr >= civet_standard_iters and itr < civet_standard_iters + civet_rampup_iters:
                    running_eps += epsilon/civet_rampup_iters
                    cur_eps = running_eps
                elif itr >= civet_standard_iters + civet_rampup_iters:
                    cur_eps = epsilon
                if cur_eps != 0:
                    x_ub = torch.clamp(x + running_eps, 0, 1)
                    x_lb = torch.clamp(x - running_eps, 0, 1)
                    outputs = model.civet_forward(torch.cat([x_ub, x_lb], 0), deltas = deltas, max_bs_depth = max_bs_depth)
                    weights = [1 - deltas[0]]
                    for di,dj in zip(deltas[:-1], deltas[1:]):
                        weights.append(di - dj)
                    for i, output in enumerate(outputs):
                        xru = output[:output.shape[0]//2]
                        xrl = output[output.shape[0]//2:]
                        total_loss += weights[i] * torch.nn.MSELoss()(torch.where(torch.abs(xru - x) > torch.abs(xrl - x), xru, xrl), x)
            elif training_type == 'civet_sabr':
                if itr < civet_standard_iters:
                    cur_eps = 0
                if itr >= civet_standard_iters and itr < civet_standard_iters + civet_rampup_iters:
                    running_eps += epsilon/civet_rampup_iters
                    cur_eps = running_eps
                elif itr >= civet_standard_iters + civet_rampup_iters:
                    cur_eps = epsilon
                if cur_eps != 0:
                    v = utils.maximum_damage_attack(model, x, device = device, step_size = pgd_step_size, epsilon = (1-tau) * epsilon, iterations = pgd_iterations)
                    x_ub = torch.clamp(x + v + running_eps * tau, 0, 1)
                    x_lb = torch.clamp(x + v - running_eps * tau, 0, 1)
                    outputs = model.civet_forward(torch.cat([x_ub, x_lb], 0), deltas = deltas, max_bs_depth = max_bs_depth)
                    weights = [1 - deltas[0]]
                    for di,dj in zip(deltas[:-1], deltas[1:]):
                        weights.append(di - dj)
                    for i, output in enumerate(outputs):
                        xru = output[:output.shape[0]//2]
                        xrl = output[output.shape[0]//2:]
                        total_loss += weights[i] * torch.nn.MSELoss()(torch.where(torch.abs(xru - x) > torch.abs(xrl - x), xru, xrl), x)
            elif training_type == 'standard':
                pass
            else:
                raise ValueError('Unknown Training Type')

            total_loss.backward()
            optimizer.step()
            itr += 1
 
        utils.save_checkpoint(model, checkpoint_dir, model_name, epoch)