import torch
import torchvision
import torch.nn.functional as F 
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import scipy
import scipy.io
import random
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import shutil
import matplotlib.pyplot as plt
from scipy import stats
import cmath


def get_dataloader(dataset, batch_size, device):
    if dataset == 'mnist':
        datasets = get_mnist(batch_size, device)
    elif dataset == 'cifar10':
        datasets = get_cifar(batch_size, device)
    elif dataset == 'fire':
        datasets = get_fire(batch_size, device)
    else:
        raise ValueError('Unknown Dataset')
    return {'train': datasets[0], 'test': datasets[1]}

def get_mnist(batch_size = 64, device = 'cuda'):
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(x_.to(device) for x_ in torch.utils.data.dataloader.default_collate(x)))
    testset = torchvision.datasets.MNIST(root='./mnist', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(x_.to(device) for x_ in torch.utils.data.dataloader.default_collate(x))) 
    return trainloader,testloader

def get_cifar(batch_size = 32, device = 'cuda'):
    transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),  
         transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(x_.to(device) for x_ in torch.utils.data.dataloader.default_collate(x)))
    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(x_.to(device) for x_ in torch.utils.data.dataloader.default_collate(x)))
    
    return trainloader, testloader
     
def print_accuracy(net, trainloader, testloader, device, test=True, eps = 0):
    loader = 0
    loadertype = ''
    if test:
        loader = testloader
        loadertype = 'test'
    else:
        loader = trainloader
        loadertype = 'train'
    correct = 0
    total = 0
    with torch.no_grad():
        for ii, data in enumerate(loader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            x_ub = images + eps
            x_lb = images - eps
            
            outputs = net(torch.cat([x_ub,x_lb], 0))
            z_hb = outputs[:outputs.shape[0]//2]
            z_lb = outputs[outputs.shape[0]//2:]
            lb_mask = torch.eye(10).cuda()[labels]
            hb_mask = 1 - lb_mask
            outputs = z_lb * lb_mask + z_hb * hb_mask
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    correct = correct / total
    print('Accuracy of the network on the', total, loadertype, 'images: ',correct)
    return correct

def latent_space_attack(mdl, image, device, step_size = 0.001, epsilon = 0.1, iterations = 40):
    v = torch.zeros_like(image)
    for _ in range(iterations):
        x = Variable(image, requires_grad = True)
        mdl.zero_grad()
        (mean, logvar), _ = mdl(x + v)
        loss = mdl.kl_divergence_loss(mean, logvar)
        loss.backward()
        v = torch.clamp(v + step_size * x.grad, -epsilon, epsilon)
    v = torch.clamp(v + image , 0, 1) - image
    return v

def maximum_damage_attack(mdl, image, device, step_size = 0.001, epsilon = 0.1, iterations = 40):
    v = torch.zeros_like(image)
    for _ in range(iterations):
        x = Variable(image, requires_grad = True)
        mdl.zero_grad()
        _, x_reconstructed = mdl(image)
        _, noised_x_reconstructed = mdl(x + v)
        loss = -(noised_x_reconstructed - x_reconstructed).norm(p=2)
        loss.backward()
        v = torch.clamp(v + step_size * x.grad, -epsilon, epsilon)
    v = torch.clamp(v + image, 0, 1) - image
    return v

def vae_pgd(mdl, image, device, step_size = 0.001, epsilon = 0.1, iterations = 40):
    v = torch.zeros_like(image)
    for _ in range(iterations):
        x = Variable(image, requires_grad = True)
        mdl.zero_grad()
        (mean, logvar), x_reconstructed = mdl(x + v)
        reconstruction_loss = mdl.reconstruction_loss(x_reconstructed, image)
        kl_divergence_loss = mdl.kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + kl_divergence_loss
        total_loss.backward()
        v = torch.clamp(v + step_size * x.grad, -epsilon, epsilon)
    v = torch.clamp(v + image, 0, 1) - image
    return v

def save_checkpoint(model, model_dir, name, epoch):
    path = os.path.join(model_dir, name)
    os.makedirs(model_dir, exist_ok=True)
    torch.save({'state': model.state_dict(), 'epoch': epoch}, path)

def compute_delta(muu, mul, sigmau, alpha, max_depth = 20):
    p = findp(muu, mul, sigmau, alpha, max_depth = max_depth)
    delta = sigmau * torch.distributions.Normal(0,1).icdf(p)
    return delta

def findp(muu, mul, sigmau, alpha, max_depth = 20):
    obj = (mul - muu)/sigmau
    return findp_bs(obj, alpha, lb = torch.ones_like(obj, device = mul.device) * alpha + 1e-8, ub = torch.ones_like(obj, device = mul.device) * (1 + alpha)/2, depth = 0, max_depth = max_depth)

def findp_bs(obj, alpha, lb, ub, depth = 0, max_depth = 10):
    mb = (lb + ub)/2
    mbo = torch.distributions.Normal(0,1).icdf(mb) + torch.distributions.Normal(0,1).icdf(mb - alpha)
    if depth > max_depth:
        ubo = torch.distributions.Normal(0,1).icdf(ub) + torch.distributions.Normal(0,1).icdf(ub - alpha)
        return ub
    return findp_bs(obj, alpha, torch.where(obj > mbo, mb, lb), torch.where(obj > mbo, ub, mb), depth + 1, max_depth = max_depth)

class FireDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True, ratio=1, idx=1, preprocessing=True):
        self.root = root
        self.train = train
        self.ratio = ratio
        self.length = int(len([name for name in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, name))])*ratio)
        self.idx = idx
        self.preprocessing = preprocessing

    def norm_processing(self, channel):
        channel = channel - np.mean(channel)
        channel = channel/np.max(np.abs(channel))
        return channel

    def norm_processing2(self, channel):
        channel = np.sqrt(self.args.ant_num)*channel/np.linalg.norm(channel, ord=2)
        return channel

    def channel_processing(self, channel):
        channel_tmp = channel[0] + channel[1] * 1j
        channel_tmp = channel_tmp / channel_tmp[0][0]
        tmp = np.angle(channel_tmp[0])

        tmp = np.unwrap(tmp)
        x = np.arange(channel.shape[-1])
        y = tmp
        slope = np.sum((x - np.mean(x))*(y - np.mean(y)))/np.sum(pow(x - np.mean(x), 2))

        phase_array = [cmath.exp(-1j*i*slope) for i in range(channel.shape[-1])]
        for i in range(channel.shape[-2]):
            channel_tmp[i] = channel_tmp[i]*phase_array
        channel_tmp = channel_tmp/np.max(np.abs(channel_tmp))
        channel[0] = np.real(channel_tmp)
        channel[1] = np.imag(channel_tmp)
        return channel

    def channel_processing_whole(self, channel_whole):
        channel = channel_whole[:, :, :self.20]
        channel_tmp = channel[0] + channel[1] * 1j
        cfo = channel_tmp[0][0]
        channel_tmp = channel_tmp / channel_tmp[0][0]
        tmp = np.angle(channel_tmp[0])

        tmp = np.unwrap(tmp)
        x = np.arange(channel.shape[-1])
        y = tmp
        slope = np.sum((x - np.mean(x))*(y - np.mean(y)))/np.sum(pow(x - np.mean(x), 2))

        phase_array = [cmath.exp(-1j*i*slope) for i in range(channel.shape[-1])]
        for i in range(channel.shape[-2]):
            channel_tmp[i] = channel_tmp[i]*phase_array
        channel[0] = np.real(channel_tmp)
        channel[1] = np.imag(channel_tmp)

        channel_2 = channel_whole[:, :, self.20:]
        channel_tmp2 = channel_2[0] + channel_2[1] * 1j
        channel_tmp2 = channel_tmp2 / cfo
        tmp = np.angle(channel_tmp2[0])
        tmp = np.unwrap(tmp)
        x = np.arange(channel.shape[-1])
        y = tmp
        slope = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum(pow(x - np.mean(x), 2))

        phase_array = [cmath.exp(-1j * i * slope) for i in range(channel.shape[-1])] 
        for i in range(channel.shape[-2]):
            channel_tmp2[i] = channel_tmp2[i] * phase_array

        channel_2[0] = np.real(channel_tmp2)
        channel_2[1] = np.imag(channel_tmp2)
        return np.concatenate((channel, channel_2), -1)

    def __getitem__(self, idx):
        idx = int(idx/self.ratio)
        label = idx
        channel = np.load(self.root+'/{}.npy'.format(idx))
        channel = channel[:, self.args.ant_offset:self.args.ant_num+self.args.ant_offset, :]
        self.channel = channel.astype('float64')

        self.channel_ul = self.channel[:, :, :self.20]
        self.channel_dl = self.channel[:, :, self.20:]
        if not self.preprocessing:
            self.channel = np.concatenate([self.channel_ul, self.channel_dl], 2)
            return self.channel, label
        self.channel_ul = channel_processing_torch_batch(self.channel_ul)
        self.channel_dl = channel_processing_torch_batch(self.channel_dl)
        self.channel = np.concatenate([self.channel_ul, self.channel_dl], 2)

        if self.args.polar:
            tmp = self.channel[0] + self.channel[1] * 1j
            tmp2 = np.zeros(self.channel.shape)
            tmp2[0] = np.angle(tmp)
            tmp2[1] = np.abs(tmp)
            self.channel = tmp2
        return self.channel, label

    def __len__(self):
        return self.length
    
def unwrap_torch(x):
        m = len(x)
        for i in range(m-1):
            val = x[i+1] - x[i]
            if torch.abs(val) > torch.pi:
                x[i+1] = x[i+1] - 2*torch.pi*torch.ceil((val-torch.pi)/(2*torch.pi))
        return x

def unwrap_torch_batch(x, device = 'cuda'):
    m = x.shape[1]
    for i in range(m - 1):
        val = x[:,i+1] - x[:,i]
        x[:,i+1] -= 2 * torch.pi * torch.ceil((val - torch.pi)/(2 * torch.pi)) * (torch.abs(val) > torch.pi)
    return x

def channel_processing_torch_batch(channel_batch, device = 'cuda'): #11.8 ms ± 507 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    channel_tmps = channel_batch[:,0] + channel_batch[:,1] * 1j
    channel_tmps = torch.einsum('ijk,i->ijk',channel_tmps,1/channel_tmps[:,0,0])

    tmps = torch.angle(channel_tmps[:,0]).to(device)
    tmps_unwrap = unwrap_torch_batch(tmps)
    x = torch.arange(channel_batch.shape[-1],dtype=torch.float64).to(device)
    x_norm = x - torch.mean(x).to(device)
    slopes = torch.sum(x_norm * (tmps_unwrap - torch.mean(tmps_unwrap)) , axis = 1)/torch.sum(torch.pow(x_norm, 2))
    phase_arrays = torch.exp(-1j*torch.einsum('i, j -> ij',slopes, x)).to(device)
    channel_tmps = torch.einsum('ijk, ik -> ijk',channel_tmps, phase_arrays)
    channel_tmps = torch.einsum('ijk, i -> ijk',channel_tmps, 1/torch.amax(torch.abs(channel_tmps), dim = (1,2)))
    channel_finals = torch.concat([torch.real(channel_tmps), torch.imag(channel_tmps)], 1).view(*channel_batch.shape)
    return channel_finals

def dataloader_to_tensor(self, dl, device = 'cuda'):
    dl_len = len(dl)
    dl_shape = next(iter(dl))[0].shape
    input_tensor = torch.zeros([dl_shape[0]*dl_len, *dl_shape[1:3], 20], dtype = torch.float64).to(device)
    output_tensor = torch.zeros([dl_shape[0]*dl_len, *dl_shape[1:3], 20], dtype = torch.float64).to(device)
    for i, (inp, _) in enumerate(dl):
        inp = inp.to(device)
        input_tensor[i * dl_shape[0] : i * dl_shape[0] + inp.shape[0], :, :, :] = inp[:,:,:,:int(20)]
        output_tensor[i * dl_shape[0] : i * dl_shape[0] + inp.shape[0], :, :, :] = self.channel_processing_torch_batch(inp[:, :, :, -int(20):])
        if inp.shape[0] != dl_shape[0]:
            input_tensor = input_tensor[:-(dl_shape[0] - inp.shape[0])]
            output_tensor = output_tensor[:-(dl_shape[0] - inp.shape[0])]
            break
    return input_tensor, output_tensor

def dataloader_to_device_storage(dl, batch_size = 32, device = 'cuda'):
    data_tensors = dataloader_to_tensor(dl, device = device)
    temp_dataset = torch.utils.data.TensorDataset(*data_tensors)
    return torch.utils.data.DataLoader(temp_dataset, shuffle = True, batch_size = batch_size, drop_last = True)

def get_fire(batch_size=64, root="./data", ratio=1, idx=1, preprocessing=True, device = 'cuda'):
    train_set = FireDataset(root=root + "/train", ratio=ratio, idx=idx, preprocessing=preprocessing)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_loader = dataloader_to_device_storage(train_loader, batch_size = batch_size, device = device)
    test_set = FireDataset(root=root + "/test", ratio=ratio, idx=idx, preprocessing=preprocessing)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    test_loader = dataloader_to_device_storage(test_loader, batch_size = batch_size, device = device)
    return train_loader, test_loader

def nulling_loss(y, x):
    bins = 20
    y = to_complex(y).cpu()
    x = to_complex(x)
    x = x.view(x.shape[0], x.shape[2], x.shape[3])
    y = y.view(y.shape[0], y.shape[2], y.shape[3])
    num_ant = 8
    loss = 0
    num = 0
    for i in range(num_ant):  # y
        for j in range(i+1, num_ant):  # x
            loss += abs((y[:, i, :]*x[:, j, :]/x[:, i, :]) - y[:, j, :])/abs(y[:, j, :])
            num = num + 1
    return torch.mean(torch.mean(loss/num, dim=1))

def to_complex(chans, polar = False):
    if not polar:
        if chans.shape[1]%2 != 0:
            print("ERROR:cannot convert odd #cols to complex")
            return None
        mid = int(chans.shape[1]/2)
        real = chans[:,:mid]
        # real = np.array(real)
        imag = chans[:,mid:]
        # imag = np.array(imag)
        complex_chan = real+1j*imag
    else:
        phase = chans[:, :1].cpu()
        abs_ = chans[:, 1:].cpu()
        # complex_chan = np.zeros(phase.shape)
        complex_chan = abs_ * (np.cos(phase) + np.sin(phase)*1j)
    return complex_chan
    
def evm_loss2(y, x):
    bins = 20
    y = to_complex(y)
    x = to_complex(x)
    # y = norm_processing2(y)
    # x = norm_processing2(x)
    x = x.view(x.shape[0], x.shape[2], x.shape[3])
    y = y.view(y.shape[0], y.shape[2], y.shape[3])
    return 10*torch.log10(torch.sum((torch.abs(y-x))**2, (1, 2))/torch.sum((torch.abs(x))**2, (1, 2)))
    
def transformation_sample(v, input_mean=0.4, mn=0, mx=2):
    """
    Generate CFO + shift delay
    """
    batch_size = v.shape[0]
    p = torch.exp(1j * torch.rand(batch_size, requires_grad=False).uniform_(mn, mx).to('cuda') * torch.pi)
    cfo = torch.einsum('ijkl,i->ijkl', torch.ones_like(v[:,0:1,:,:], requires_grad=False).to('cuda'), p).squeeze()
    shift_p = torch.randint(80, (batch_size, ), requires_grad=False).to('cuda')/bw
    tmp = -1j * 2*torch.pi * torch.from_numpy(shift_p[:20]).to('cuda')
    shift_delay = torch.exp(torch.einsum('i, j->ij', shift_p, tmp))
    shift_delay_antennas = shift_delay[:, None, :].repeat([1, 4, 1])

    g_noise = torch.normal(0, 0.1, v.size()).to('cuda')/0.8 * 1 * input_mean
    # return torch.rand(1).uniform_(0.5, 1)*cfo * shift_delay_antennas, g_noise
    return cfo * shift_delay_antennas, g_noise
    
def channel_fading(v, h_attack):
    """
    input size: v: [Batchsize, 2, attacker ant, bins], e.g., [256, 2, 1, 52]
                h_attack: [1, 2, 4, 52]
    return size: [Batchsize, 2, 4, bins]
    """
    batch_size = v.shape[0]
    #v_faded_final = torch.ones([batch_size, h_attack.shape[1],  h_attack.shape[2], h_attack.shape[3]]).to(device)
    v_faded_final = v
    v_complex = v[:, 0] + v[:, 1]*1j
    h_attack_complex = h_attack[:, 0] + h_attack[:, 1]*1j
    v_faded = v_complex * h_attack_complex
    v_faded_final[:, 0], v_faded_final[:, 1] = torch.real(v_faded).to('cuda'), torch.imag(v_faded).to('cuda')
    return v_faded_final

def complex_multi(a, b):
    if a.size()[1] != 2:
        raise ValueError
    a1 = a[:, 0] + 1j * a[:, 1]
    r = a1 * b
    r_r = torch.view_as_real(r).permute([0, 3, 1, 2])
    return r_r


def firepgd(self, v_all, random_seeds, h_attack, input_mean, data_input, epsilon_ratio, gt, lr, transforms = 4):
    dummy_input = next(iter(self.train_x))[0]
    batch_size = data_input.shape[0]
    v = torch.zeros([random_seeds, dummy_input.shape[1], 1, dummy_input.shape[3]], device = 'cuda', requires_grad = True)
    success_rate = 0
    if transforms == 0:
        x, g_noise = 1, 0
        transforms = 1
    else:
        x, g_noise = transformation_sample(channel_fading(v.repeat(transforms, 1, 4, 1), h_attack), input_mean=input_mean)

    for i_pgd in range(20):
        v_1 = channel_fading((v_all + v).repeat(transforms, 1, 4, 1), h_attack)
        v_ = complex_multi(v_1, x) + g_noise
        input_after = channel_processing_torch_batch(data_input.repeat(random_seeds * transforms, 1, 1, 1) + v_.repeat(batch_size, 1, 1, 1))
        y_pred, mu, var = self.net(input_after, self.state)
        evmloss_tst = evm_loss2(y_pred, gt.repeat(random_seeds * transforms, 1, 1, 1))
        success_rate = torch.sum(evmloss_tst > 14) / len(evmloss_tst)

        recon_loss = 0
        for i in range(random_seeds):
            recon_loss += evm_loss2(y_pred.reshape(random_seeds, batch_size * transforms, *y_pred.shape[1:])[i], gt.repeat(transforms, 1, 1, 1)).sum()
        (-recon_loss).backward()
        grad = v.grad
        v.data = (v + lr * grad/grad.mean()).clamp(-epsilon_ratio, epsilon_ratio)
        v.grad.zero_()
        if success_rate > 0.8:
            break
    return v

def comp_success_rate(net, v, random_seeds, h_attack, input_mean, data_input, state, gt):
    batch_size = data_input.shape[0]
    state = net.init_hidden(batch_size)
    with torch.no_grad():
        v_1 = channel_fading(v.repeat(1, 1, 4, 1), h_attack)
        x_pre, g_noise = transformation_sample(v_1, input_mean=input_mean)
        v_pre = complex_multi(v_1, x_pre) + g_noise
        input_after = channel_processing_torch_batch(data_input.repeat(random_seeds, 1, 1, 1) + v_pre.repeat(batch_size, 1, 1, 1))
        (mu, var), y_pred = net(input_after, state)
        evmloss_tst = evm_loss2(y_pred, gt.repeat(random_seeds, 1, 1, 1))
        success_rate = torch.sum((evmloss_tst > 0.9).reshape(batch_size, random_seeds), axis = 0)/batch_size
    return success_rate

def rafaevaluate(net, state, dataset, v, input_mean, h_attack, random_seeds = 1):
    evm_all = torch.zeros(len(dataset), random_seeds)
    real_sr_all = torch.zeros(len(dataset), random_seeds)
    success_rate_all = torch.zeros(len(dataset), random_seeds)
    with torch.no_grad():
        num_all = 0
        num_success = 0
        for i, (data_input, gt) in enumerate(dataset):
            v_ = channel_fading(v.repeat(len(data_input), 1, 4, 1), h_attack)

            x, g_noise = transformation_sample(v_, input_mean=input_mean)
            v_all_test_2 = complex_multi(v_.data, x) + g_noise
            input_after = channel_processing_torch_batch(data_input.repeat(random_seeds, 1, 1, 1) + v_all_test_2)
            y_pred, mu, var = net(input_after, state)
            evmloss_trn = evm_loss2(y_pred, gt.repeat(random_seeds, 1, 1, 1))

            evm_all[i] = torch.mean(evmloss_trn.reshape(data_input.shape[0], random_seeds), axis = 0)
            real_sr_all[i] = torch.mean(torch.abs(v_all_test_2).reshape(data_input.shape[0], random_seeds, -1), axis = (0, 2)) / torch.mean(torch.abs(data_input))
            success_rate_all[i] = torch.sum((evmloss_trn > 0.9).reshape(-1, random_seeds), axis = 0)/(data_input.shape[0] * len(dataset))
    return torch.mean(evm_all, axis = 0), torch.median(evm_all, axis = 0).values, torch.sum(success_rate_all, axis = 0), torch.mean(real_sr_all, axis = 0)

def RAFA(net, dataset, random_seeds = 1, warm_start = None, h_attack=None, lr = 0.001, epochs = 10, alpha = 1, percent_attack = 0.5, epsilon = 0.25, rate_thre = 0.9):
    h_attack = np.load('latest.npy',allow_pickle=True).item()
    if len(h_attack['csi'][1]) == 4:
        h_attack = np.array(h_attack['csi'][1])[:, 4][:, :20]
    else:
        h_attack = np.array(h_attack['csi'])[:, 4][:, :20]
    h_attack = torch.from_numpy(h_attack).to('cuda')
    h_attack = h_attack / torch.mean(torch.abs(h_attack), 1, keepdim=True)
    h_attack = torch.concat((torch.real(h_attack)[None, :], torch.imag(h_attack)[None, :]), 0)
    dummy_input = next(iter(dataset))[0]
    batch_size = len(dummy_input)
    input_mean = dummy_input.abs().mean()
    state = net.init_hidden(batch_size)
    epsilon_ratio = epsilon * input_mean
    if warm_start is None:
        v_all = torch.zeros([random_seeds, dummy_input.shape[1], 1, dummy_input.shape[3]], device = 'cuda').uniform_(-epsilon_ratio/10, epsilon_ratio/10)
    else:
        v_all = torch.clone(warm_start).repeat(random_seeds, 1, 1, 1).to('cuda')
    for i_episode in range(epochs):
        evm_all = []
        evm_tst_all = []
        for i, (data_input, gt), in enumerate(dataset):
            if i/len(dataset) > percent_attack:
                break
            net.zero_grad()
            mask = comp_success_rate(v_all, random_seeds, h_attack, input_mean, data_input, self.state, gt) < rate_thre
            v = firepgd(v_all[mask], sum(mask), h_attack, input_mean, data_input, epsilon_ratio, gt, lr)
            v_all[mask] = (v_all[mask] + v.data).clamp(-epsilon_ratio, epsilon_ratio)

        mean, median, asr, real_sr = rafaevaluate(net, state, dataset, v_all, input_mean, h_attack, random_seeds = random_seeds)
        best_idx = torch.argmax(mean)
        lr *= alpha
        v_all[0] = v_all[best_idx]
        v_all[1:] = v_all[0].repeat(random_seeds - 1, 1, 1, 1) + torch.zeros([random_seeds - 1, dummy_input.shape[1], 1, dummy_input.shape[3]], device = 'cuda').uniform_(-lr, lr)
    return v_all[0:1]