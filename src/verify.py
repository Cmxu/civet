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


def baseline(model, dataloader, device):
    data_stream = tqdm(dataloader)
    average_loss = 0
    model.unrobust()
    count = 0
    for (x, _) in data_stream:
        _, x_r = model(x)
        loss = torch.nn.MSELoss()(x_r, x)
        average_loss += loss
        count += 1
    return average_loss/count

def verify_mda(model, dataloader, device, step_size = 0.001, epsilon = 0.0, iterations = 40):
    data_stream = tqdm(dataloader)
    average_loss = 0
    model.unrobust()
    count = 0
    for (x, _) in data_stream:
        v = utils.maximum_damage_attack(model, x, device = device, step_size = step_size, epsilon = epsilon, iterations = iterations)
        _, x_rv = model(x + v)
        loss = torch.nn.MSELoss()(x_rv, x)
        average_loss += loss
        count += 1
    return average_loss/count

def verify_lsa(model, dataloader, device, step_size = 0.001, epsilon = 0.0, iterations = 40):
    data_stream = tqdm(dataloader)
    average_loss = 0
    model.unrobust()
    count = 0
    for (x, _) in data_stream:
        v = utils.latent_space_attack(model, x, device = device, step_size = step_size, epsilon = epsilon, iterations = iterations)
        _, x_rv = model(x + v)
        loss = torch.nn.MSELoss()(x_rv, x)
        average_loss += loss
        count += 1
    return average_loss/count

def verify_rafa(model, dataloader, device, step_size = 0.001, epsilon = 0.0, iterations = 40):
    data_stream = tqdm(dataloader)
    average_loss = 0
    model.unrobust()
    count = 0
    for (x, _) in data_stream:
        v = utils.RAFA(model, x, device = device, lr = step_size, epsilon = epsilon, epochs = iterations)
        _, x_rv = model(x + v)
        loss = utils.evm_loss2(x_rv, x)
        average_loss += loss
        count += 1
    return average_loss/count

def evm_baseline(model, dataloader, device):
    data_stream = tqdm(dataloader)
    average_loss = 0
    model.unrobust()
    count = 0
    for (x, _) in data_stream:
        _, x_v = model(x)
        loss = utils.evm_loss2(x_v, x)
        average_loss += loss
        count += 1
    return average_loss/count

def verify_lagrangian(model, dataloader, device, epsilon):
    data_stream = tqdm(dataloader)
    muus, muls, sigmaus, sigmals = [], [], [], []
    model.robust()
    for (x, _) in data_stream:
        xu = torch.clamp(x + epsilon, 0, 1)
        xl = torch.clamp(x - epsilon, 0, 1)
        x = torch.cat([xu, xl], 0)
        muu, mul, sigmau, sigmal = model.ibp_latent(x)
        muus.append(muu)
        muls.append(mul)
        sigmaus.append(sigmau)
        sigmals.append(sigmal)
    muus = torch.concat(muus, axis = 0)
    muls = torch.concat(muls, axis = 0)
    sigmaus = torch.concat(sigmaus, axis = 0)
    sigmals = torch.concat(sigmals, axis = 0)
    return torch.stack([muus, muls, sigmaus, sigmals])

def verify(model, dataloader, device, step_size, epsilon, iterations):
    model.eval()
    baseline_perf = baseline(model, dataloader, device)
    mda_perf = verify_mda(model, dataloader, device, step_size = step_size, epsilon = epsilon, iterations = iterations)
    lsa_perf = verify_lsa(model, dataloader, device, step_size = step_size, epsilon = epsilon, iterations = iterations)
    ibplatent = verify_lagrangian(model, dataloader, device, epsilon)
    return baseline_perf, mda_perf, lsa_perf, ibplatent

def verify_fire(model, dataloader, device, step_size, epsilon, iterations):
    model.eval()
    baseline_perf = evm_baseline(model, dataloader, device)
    rafa_perf = verify_rafa(model, dataloader, device, step_size = step_size, epsilon = epsilon, iterations = iterations)
    ibplatent = verify_lagrangian(model, dataloader, device, epsilon * 100)
    return baseline_perf, rafa_perf, ibplatent

