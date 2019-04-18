__author__ = 'Yimeng'

import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from tensorboardX import SummaryWriter

from loading_data import DataLoader

_BATCH_SIZE = 10

def visualizer(_input, _output):
    _index = np.random.randint(0,_output.shape[0])
    input_map = _input[_index]
    output_map = _output[_index]
    plt.figure(figsize=(20,5))
    ax1 = plt.subplot2grid((2,2),(0,0))
    sns.heatmap(input_map[0])
    ax1.set_title('Masked input submap')
    ax2 = plt.subplot2grid((2,2),(0,1))
    sns.heatmap(output_map[0])
    ax2.set_title('Input mask')
    ax3 = plt.subplot2grid((2,2),(1,0))
    sns.heatmap(input_map[1])
    ax3.set_title('Masked output submap')
    ax4 = plt.subplot2grid((2,2),(1,1))
    sns.heatmap(input_map[1])
    ax4.set_title('Output mask')
    plt.show()

def _convert_plot_to_image(figure):
    figure.canvas.draw()
    data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    return data

def _tbX_visualizer(_input, _output, w, epoch):
    _index = np.random.randint(0,_output.shape[0])
    input_map = _input[_index]
    output_map = _output[_index]
    fig = plt.figure(figsize=(20,5))
    ax1 = plt.subplot2grid((2,2),(0,0))
    sns.heatmap(input_map[0])
    ax1.set_title('Masked input submap')
    ax2 = plt.subplot2grid((2,2),(0,1))
    sns.heatmap(output_map[0])
    ax2.set_title('Input mask')
    ax3 = plt.subplot2grid((2,2),(1,0))
    sns.heatmap(input_map[1])
    ax3.set_title('Masked output submap')
    ax4 = plt.subplot2grid((2,2),(1,1))
    sns.heatmap(input_map[1])
    ax4.set_title('Output mask')
    w.add_image('intermediate_plots', _convert_plot_to_image(fig), global_step = epoch, dataformats='HWC')

class BasicConv(nn.Module):
    def __init__(self,_in_channels, _out_channels, kernel_size,isPooling=True,isUpsampling=False):
        super(BasicConv,self).__init__()
        self.isPooling = isPooling
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(_in_channels,_out_channels,kernel_size)
        self.relu = nn.ReLU(True)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.up_sample = None # empty function
        if isUpsampling:
            self.up_sample = nn.Upsample(scale_factor=2.0, mode='bilinear') # define function
        

    def forward(self,x):
        paddingSize = int((self.kernel_size-1)/2)
        if self.up_sample is not None:
            x = self.up_sample(x)
        x = self.conv(x)
        y = F.pad(x, pad=[paddingSize,paddingSize,paddingSize,paddingSize])
        y = self.relu(y)
        if self.isPooling:
            y = self.maxpool2d(y)
        return y

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder_conv = nn.Sequential(
            BasicConv(2,16,7),
            BasicConv(16,32,5),
            BasicConv(32,64,3))
        self.encoder_output = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(128*128, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            nn.Linear(512,256),
            nn.ReLU(True))
        self.decoder_input = nn.Sequential(
            # nn.Dropout(p=0.1),
            nn.Linear(256,512),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(512,128*128),
            nn.ReLU(True))
        self.decoder_deconv = nn.Sequential(      
            BasicConv(64,32,3,isPooling=False,isUpsampling=True),
            BasicConv(32,16,5,isPooling=False,isUpsampling=True),
            BasicConv(16,2,7,isPooling=False,isUpsampling=True))
    def forward(self,x):
        x = self.encoder_conv(x)
        # print(x.shape)
        x = x.view(x.size(0),-1) # prepare for FC
        # print(x)
        # print(x.shape)
        x = self.encoder_output(x)
        x = self.decoder_input(x)
        # print(x.shape)
        x = x.view(x.size(0),64,16,16)
        # print(x.shape)
        x = self.decoder_deconv(x)
        return x

warnings.filterwarnings("ignore")

DL = DataLoader()
tf_writer = SummaryWriter()
# trainloader = []
# for i in range(DL.training_data.size):
#     trainloader.append(torch.from_numpy(DL.training_data[i]))
trainloader = torch.utils.data.DataLoader(torch.from_numpy(np.array(list(DL.training_data))), batch_size=_BATCH_SIZE,
                                                      shuffle=True)
testloader = torch.utils.data.DataLoader(torch.from_numpy(np.array(list(DL.validation_data))), batch_size=1,
                                                      shuffle=True)
num_epochs =500

model = Autoencoder().cuda()
distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)

training_dataset_length = len(trainloader.dataset)

# f = open('result','w')
for epoch in range(num_epochs):
    totalLoss = 0
    for _data in trainloader:
        data = _data[:,:,0:-1,0:-1]
        data = Variable(data).cuda()
        data = data.float()
        # ===================forward=====================
        # data=data[:,None]
        output = model(data)
        loss = distance(output, data)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totalLoss += loss
    # ===================log========================
    #visualizer(data.numpy(),output.data.numpy())
    tf_writer.add_scalar('data/loss', totalLoss/training_dataset_length, epoch)
    # f.writelines('epoch [{}/{}], loss:{:.4f}\n'.format(epoch+1, num_epochs, loss.data))
    if epoch % 10 == 0:
        _tbX_visualizer(_data.cpu().data.numpy(),output.cpu().data.numpy(),tf_writer,epoch)
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data))
tf_writer.close()
# f.close()
torch.save(model.state_dict(),'trained_autoencoder.pth')

# model = Autoencoder().cpu()
# model.load_state_dict(torch.load('trained_autoencoder_batch10_01.pth',map_location='cpu'))  
def validator(testloader,model):
    for data in testloader:
        data = Variable(data).cpu()
        data = data.float()
        # ===================forward=====================
        # data=data[:,None]
        output = model(data)
        visualizer(data.numpy(),output.data.numpy())
