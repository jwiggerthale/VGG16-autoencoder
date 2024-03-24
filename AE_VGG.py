'''
Implementation of covolutional autoencoder following VGG16 architecture
Only convolutional layers of VGG16 architecture are used
Implemented in three models 
  VGG16_Encoder 
  VGG16_Decoder 
  Autoencoder (puts encoder and decoder together)
IMPORTANT: Only works with input where size of each dimension is 2^n where n is at least 6
'''
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

'''
Encoder leanrs reduced representation for the data 
IMPORTANT: Adapt number of inputs to first fully connected layer to number of features you expect
    --> last convolutional layer yields 512 channels where every dimension of the input is by factor 2^5 smaller than the original input
Forwrad pass returns bottle neck (reduced representation) and pooling indices 
'''
class VGG16_Encoder(nn.Module):
    def __init__(self):
        super(VGG16_Encoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.pooling1 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices = True)
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.pooling2 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices = True)
        self.encoder5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.encoder6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.encoder7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.pooling3 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices = True)
        self.encoder8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.encoder9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.encoder10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.pooling4 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices = True)
        self.encoder11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.encoder12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.encoder13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.pooling5 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices = True)
       
        
    def forward(self, x):
        pooling_indices = []
        out = self.encoder1(x)
        out = self.encoder2(out)
        out =self.pooling1(out)
        pooling_indices.append(out[1])
        out = out[0]
        out = self.encoder3(out)
        out = self.encoder4(out)
        out =self.pooling2(out)
        pooling_indices.append(out[1])
        out = out[0]
        out = self.encoder5(out)
        out = self.encoder6(out)
        out = self.encoder7(out)
        out =self.pooling3(out)
        pooling_indices.append(out[1])
        out = out[0]
        out = self.encoder8(out)
        out = self.encoder9(out)
        out = self.encoder10(out)
        out =self.pooling4(out)
        pooling_indices.append(out[1])
        out = out[0]
        out = self.encoder11(out)
        out = self.encoder12(out)
        out = self.encoder13(out)
        out =self.pooling5(out)
        pooling_indices.append(out[1])
        out = out[0]

        return out, pooling_indices
    
'''
Decoder reconstructs original data from reduced representation
IMPORTANT: Adapt number of outputs of last fully connected layer to size of the first convolutional layer
           Adapt shape of input to first convolutional layer to shape you expect
Forward pass returns reconstructed data
'''    
class VGG16_Decoder(nn.Module):
    def __init__(self):
        super(VGG16_Decoder, self).__init__()
        self.unpooling5 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
        self.decoder13 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.decoder12 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.decoder11 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.unpooling4 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
        self.decoder10 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.decoder9 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.decoder8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.unpooling3 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
        self.decoder7 = nn.Sequential( 
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.decoder6 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.unpooling2 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
        self.decoder4 = nn.Sequential( 
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.unpooling1 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1))
       
        
    def forward(self, bn, pooling_indices):
        reconstructed = self.unpooling5(bn, indices = pooling_indices[4])
        reconstructed = self.decoder13(reconstructed)
        reconstructed = self.decoder12(reconstructed)
        reconstructed = self.decoder11(reconstructed)
        reconstructed = self.unpooling4(reconstructed, pooling_indices[3])
        reconstructed = self.decoder10(reconstructed)
        reconstructed = self.decoder9(reconstructed)
        reconstructed = self.decoder8(reconstructed)
        reconstructed = self.unpooling3(reconstructed, pooling_indices[2])
        reconstructed = self.decoder7(reconstructed)
        reconstructed = self.decoder6(reconstructed)
        reconstructed = self.decoder5(reconstructed)
        reconstructed = self.unpooling2(reconstructed, pooling_indices[1])
        reconstructed = self.decoder4(reconstructed)
        reconstructed = self.decoder3(reconstructed)
        reconstructed = self.unpooling1(reconstructed, pooling_indices[0])
        reconstructed = self.decoder2(reconstructed)
        reconstructed = self.decoder1(reconstructed)

        return reconstructed

'''
Combines encoder and decoder
Returns bottle neck and reconstructed data
'''
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.Encoder = VGG16_Encoder()
        self.Decoder = VGG16_Decoder()
        
    def forward(self, x):
        bn, pooling_indices = self.Encoder.forward(x)
        reconstructed = self.Decoder.forward(bn, pooling_indices)
        return bn, reconstructed
        
