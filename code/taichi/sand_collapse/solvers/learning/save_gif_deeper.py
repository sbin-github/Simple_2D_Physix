import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from CoordConvModule import *

from matplotlib import pyplot as plt
from celluloid import Camera
import numpy as np
from random import randrange
# from torchsummary import summary

testset = torch.load('../data/testset_10steps_2.pt')
testloader = DataLoader(testset, batch_size=1, shuffle=True)

width_fig = 100

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 100, 100]
        # Output size: [batch, 3, 100, 100]
        self.encoder = nn.Sequential(
            CoordConv(3, 64, kernel_size = 1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size = 5, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 7, stride=1, padding=1),
            # nn.BatchNorm2d(6
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 7, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, 5, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 3, 1, stride=1)
        )

    def forward(self, x):
        # print(x.size())
        # wall = x[:, -1]
        # print(wall.size())
        x = self.encoder(x)
        x = self.decoder(x)
        # print(x.size())
        # x = torch.cat((x, wall.reshape(len(x), 1, width_fig, width_fig)), 1)
        # print(x.size())
        return x
        
class StepAE(nn.Module):
    def __init__(self):
        """Variational Auto-Encoder Class"""
        super(StepAE, self).__init__()
        
        self.enc = Autoencoder()

    def step(self, x): # For actual run after training
        x = self.enc(x)
        return x

    def forward(self, x): # For training
        # Encode x to z
        recon1 = self.enc(x)
        recon2 = self.enc(recon1)
        recon3 = self.enc(recon2)
        recon4 = self.enc(recon3)
        # recon5 = self.enc(recon4)
        
        return recon1, recon2, recon3, recon4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_origin = StepAE().to(device)
model_origin.load_state_dict(torch.load("latest_deeper_net.pt"))
model_origin.eval()

def animation(figname):
    model_origin.eval()
    dataiter = iter(testloader)
    images, *_ = dataiter.next()
    del _
    
    tmpmtx_cplot = [[[0, 0.5, 0.5] for i in range(width_fig)] for i in range(width_fig)]
    tmpmtx_cplot_mul = [[[2, 10, 10] for i in range(width_fig)] for i in range(width_fig)]
    recon = model_origin.step(images.permute(0, 3, 1, 2).cuda())

    # del _
    torch.cuda.empty_cache()
    # get sample outputs

    steps = 200
    skip = 2
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    camera1 = Camera(fig1)
    camera2 = Camera(fig2)

    output = recon.detach().view(1, 3, 100, 100).cpu()[0].permute(1,2,0)
    ax1.imshow(np.multiply(output.numpy(), tmpmtx_cplot_mul) + tmpmtx_cplot, vmax=50, vmin=-50)
    camera1.snap()
    mass = recon.detach().view(1, 3, 100, 100).cpu()[0][0].view(1, 100, 100).permute(1,2,0)
    ax2.imshow(mass, vmax=5, vmin=0, cmap='gray_r')
    camera2.snap()
    
    ii = 0
    for i in range(steps):
        recon = model_origin.step(recon)
        # del _
        
        if i%skip == 0:
            ii+=1
            # ax = plt.subplot(12, 1, ii + 2)
            # print(i)
            # output is resized into a batch of iages
            # use detach when it's an output that requires_grad
            output = recon.detach().view(1, 3, 100, 100).cpu()[0].permute(1, 2, 0)
            mass = recon.detach().view(1, 3, 100, 100).cpu()[0][0].view(1, 100, 100).permute(1,2,0)
            # print(output.min(), output.max())
            # plt.imshow(output)
            ax1.imshow(np.multiply(output.numpy(), tmpmtx_cplot_mul) + tmpmtx_cplot, vmax=20, vmin=-20)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            camera1.snap()

            ax2.imshow(mass, vmax=5, vmin=0, cmap='gray_r')
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
            camera2.snap()
            del output
            torch.cuda.empty_cache()
    # plt.savefig("../data/model/"+timestampStr+"/fig_{}_{}.png".format(epochnow, step))
    # torch.save(images, "../data/model/"+timestampStr+"/input_{}_{}.pt".format(epochnow,step))
    animation = camera1.animate()
    animation.save("../data/gif_deeper/"+figname + '.gif', writer = 'imagemagick', fps=100)
    animation2 = camera2.animate()
    animation2.save("../data/gif_deeper/"+figname + '_mass.gif', writer = 'imagemagick', fps=100)
    del images
    del recon
    del dataiter


if __name__ == '__main__':
    for i in range(1, 1001):
        animation(figname="gif_{}".format(int(i)))