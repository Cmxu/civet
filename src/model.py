import torch
from torch.autograd import Variable
from torch import nn
from ibp_modules import *
from utils import *


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

    def forward(self, x):
        encoded = self.encoder(x)
        mean, logvar = self.q(encoded, x.shape[0])
        z = self.z(mean, logvar)
        z_projected = self.project(z)
        x_reconstructed = self.decoder(z_projected)
        return (mean, logvar), x_reconstructed.view(x.shape)

    def civet_forward(self, x, deltas = [0.05], max_bs_depth = 20):
        encoded = self.encoder(x)
        mean, logvar = self.q(encoded, x.shape[0])
        muu = mean[:mean.shape[0]//2]
        mul = mean[mean.shape[0]//2:]
        std = logvar.mul(0.5).exp_()
        sigmau = std[std.shape[0]//2:]
        outs = []
        for delta in deltas:
            xi = compute_delta(muu, mul, sigmau, 1 - delta, max_depth = max_bs_depth)
            zu = muu + xi
            zl = mul - xi
            z = torch.cat([zu, zl], 0)
            z_projected = self.project(z)
            x_reconstructed = self.decoder(z_projected).view(x.shape)
            outs.append(x_reconstructed)
        return outs

    def ibp_latent(self, x):
        encoded = self.encoder(x)
        mean, logvar = self.q(encoded, x.shape[0])
        muu = mean[:mean.shape[0]//2]
        mul = mean[mean.shape[0]//2:]
        std = logvar.mul(0.5).exp_()
        sigmau = std[std.shape[0]//2:]
        sigmal = mean[mean.shape[0]//2:]
        return muu, mul, sigmau, sigmal
        

    def q(self, encoded, batch_size):
        unrolled = encoded.view(batch_size, -1)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def z(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(torch.randn(std.size())).to(device = std.device)
        return eps.mul(std).add_(mean)

    def reconstruction_loss(self, x_reconstructed, x):
        return nn.functional.binary_cross_entropy(x_reconstructed, x)
    
    def MSE_loss(self, x_reconstructed, x):
        return nn.MSELoss()(x_reconstructed, x)

    def kl_divergence_loss(self, mean, logvar):
        return - 0.5 * torch.sum(1+ logvar - mean.pow(2) - logvar.exp())
    
    def unrobust(self):
        if isinstance(self.q_mean, RobustModule):
            self.q_mean.unrobust()
        if isinstance(self.q_logvar, RobustModule):
            self.q_mean.unrobust()
        if isinstance(self.projection, RobustModule):
            self.q_mean.unrobust()
        for layer in self.encoder:
            if isinstance(layer, RobustModule):
                layer.unrobust()
        for layer in self.decoder:
            if isinstance(layer, RobustModule):
                layer.unrobust()

    def robust(self):
        if isinstance(self.q_mean, RobustModule):
            self.q_mean.robust()
        if isinstance(self.q_logvar, RobustModule):
            self.q_mean.robust()
        if isinstance(self.projection, RobustModule):
            self.q_mean.robust()
        for layer in self.encoder:
            if isinstance(layer, RobustModule):
                layer.robust()
        for layer in self.decoder:
            if isinstance(layer, RobustModule):
                layer.robust()
        

class ConvVAE(VAE):
    def __init__(self, z_size = 32, image_size = 28, channel_num = 1, kernel_num = 64, ibp = True):
        super(ConvVAE, self).__init__()
        conv, convt, linear = nn.Conv2d, nn.ConvTranspose2d, nn.Linear
        if ibp:
            conv, convt, linear = RobustConv2d, RobustConv2dTranspose, RobustLinear
        self.z_size = z_size
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.encoder = nn.Sequential(
            conv(channel_num, kernel_num // 4, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            conv(kernel_num // 4, kernel_num // 2, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            conv(kernel_num // 2, kernel_num, kernel_size=5, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            convt(kernel_num, kernel_num // 2, kernel_size=5, stride=2, padding=1, output_padding=int(self.image_size == 28)),
            nn.ReLU(),
            convt(kernel_num // 2, kernel_num // 4, kernel_size=5, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            convt(kernel_num // 4, channel_num, kernel_size=5, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
        x_test = torch.zeros(1, channel_num, image_size, image_size)
        e_out = self.encoder(x_test)
        self.feature_size = e_out.shape[-1]
        self.feature_volume = kernel_num * (self.feature_size ** 2)
        self.q_mean = linear(self.feature_volume, self.z_size)
        self.q_logvar = linear(self.feature_volume, self.z_size)
        self.projection = linear(self.z_size, self.feature_volume)
    
    def project(self, z):
        z = self.projection(z)
        return z.view(-1, self.kernel_num, self.feature_size, self.feature_size)

class FCVAE(VAE):
    def __init__(self, z_size = 10, image_size = 28, channel_num = 1, kernel_num = 512, ibp = True):
        super(FCVAE, self).__init__()
        linear = nn.Linear
        if ibp:
            linear = RobustLinear
        self.z_size = z_size
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.encoder = nn.Sequential(
            nn.Flatten(),
            linear(channel_num * image_size * image_size, kernel_num),
            nn.ReLU(),
            linear(kernel_num, kernel_num),
        )
        self.decoder = nn.Sequential(
            linear(kernel_num, kernel_num),
            nn.ReLU(),
            linear(kernel_num, channel_num * image_size * image_size),
            nn.Sigmoid(),
        )
        self.q_mean = linear(kernel_num, z_size)
        self.q_logvar = linear(kernel_num, z_size)
        self.projection = linear(z_size, kernel_num)
    
    def project(self, z):
        return self.projection(z)
    
class FIREVAE(VAE):
    def __init__(self, z_size = 10, image_size = 28, channel_num = 1, kernel_num = 1024, ibp = True):
        super(FIREVAE, self).__init__()

        super(FCVAE, self).__init__()
        linear = nn.Linear
        if ibp:
            linear = RobustLinear
        self.z_size = z_size
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.encoder = nn.Sequential(
            nn.Flatten(),
            linear(2 * 8 * 20, kernel_num),
            nn.LeakyReLU(),
            nn.Linear(kernel_num, kernel_num // 2),
            nn.LeakyReLU(),
            nn.Linear(kernel_num // 2, kernel_num // 4),
            nn.LeakyReLU(),
            nn.Linear(kernel_num // 4, kernel_num // 8),
            nn.LeakyReLU(),
            nn.Linear(kernel_num // 8, kernel_num // 16),
            nn.LeakyReLU(),
            nn.Linear(kernel_num // 16, self.z_size),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.z_size, kernel_num // 16),
            nn.LeakyReLU(),
            nn.Linear(kernel_num // 16, kernel_num // 8),
            nn.LeakyReLU(),
            nn.Linear(kernel_num // 8, kernel_num // 4),
            nn.LeakyReLU(),
            nn.Linear(kernel_num // 4, kernel_num // 2),
            nn.LeakyReLU(),
            nn.Linear(kernel_num // 2, kernel_num),
            nn.LeakyReLU(),
            nn.Linear(kernel_num, 2*8 * 20),
            nn.Tanh()
        )
        self.q_mean = linear(kernel_num, z_size)
        self.q_logvar = linear(kernel_num, z_size)
        self.projection = linear(z_size, z_size)
    
    def project(self, z):
        return self.projection(z)