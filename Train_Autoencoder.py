'''
This code implements training of convolutional autoencoder in PyTorch
	--> Example uses grayscale images with 1024 x 2048 pixels
In the example, enocder following VGG16 architectire is used (c.f. https://arxiv.org/pdf/1409.1556.pdf, implemented from scratch in module vgg)
  -->Decoder exactly opposite model
BOTH MODELS ONLY USE CONVOLUTIONAL LAYERS OF VGG16 ARCHITECTURE
'''


import numpy as np
import os
from PIL import Image
import copy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.models
import AE_VGG
import csv

'''
Define transformations for training-, test- and validation images 
Transformations for validation- and test images identical 
Adapt if required
'''
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.09523337495093247],[0.1498455750945764])
    ])
transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.09523337495093247],[0.1498455750945764])
    ])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.09523337495093247],[0.1498455750945764])
    ])

'''
Get image paths
Make sure, file exists 
Adapt reshaping to your specific file structure
'''
with open('/workspace/image_recognition_apis/im_paths.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)  
im_paths  = np.array(data).reshape(len(data[0])) 

'''
Split paths in those for training, validation and testing
Shuffle previously to ensure balanced distribution
'''
np.random.shuffle(im_paths)
train_paths, rest = np.split(im_paths, [int(0.8*len(im_paths))])
test_paths, val_paths = np.split(rest, [int(0.5*len(rest))])

'''
Class image dataset for loading images 
Create with: 
	File_Name: list of filepaths to images 
	transform: Instance of transforms.Compose (optional)
Methods: 
	getitem --> Returns image for index
		--> If transforms is not None, image is transformed by transforms
	len --> Returns number of image paths in filepaths
'''
class ImageDataset(Dataset):
    def __init__(self, File_Name, transform=False):
        self.image_paths = File_Name
        self.transform = transform

    def __len__(self):
        return(len(self.image_paths))


    def __getitem__(self, idx):
        im_path = '/workspace/'+self.image_paths[idx].replace('\'', '')
        #im_path = im_path.replace('\\', '/')
        #im_path = im_path.replace('tiff', 'csv')
        #im_path = im_path.replace('2023APIS2', 'Ims_APIS')
        im = Image.open(im_path)
        im = np.array(im)
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB
        if(self.transform is not None):
            im = self.transform(im)#['Image']
        return im


'''
Create instances of ImageDataset for training-, test- and validation data
Put them in a DataLoader
'''
train_data = ImageDataset(train_paths, transform_train)
test_data = ImageDataset(test_paths, transform_test)
val_data = ImageDataset(val_paths, transform_val)

train_loader = DataLoader(train_data, batch_size =8, shuffle = True, num_workers = 0, drop_last = True)
test_loader = DataLoader(test_data, batch_size = 8, shuffle = False, num_workers = 0, drop_last = True)
val_loader = DataLoader(val_data, batch_size=8, shuffle = True, num_workers=0, drop_last = True)

'''
Function validation
Use for validation of model's performance (i.e. after every epoch of training)
Call with: 
	model: Model you are currently training
	val_loader: Data loader for validation images
	criterion: Loss function you apply
Returns: 
	Average loss on validation images
'''
def validation(model: nn.Module, 
              val_loader: DataLoader,
              criterion: nn.Module, 
              device: str = 'cpu'):
    model.eval()
    val_loss = 0
    for ims in iter(val_loader):
        ims = ims.to(device)
        bn, rec = model.forward(ims)
        loss = criterion(rec, ims)
        val_loss += loss.item()
    val_loss /= len(val_loader)
    return val_loss


'''
Function for training model 
Conducts training and validation of model iteratively 
Call with: 
	model: Model you want to train 
	optimizer: Optimizer you apply
	loss_function: loss function you apply
	epoch: Number of epochs for training
	threshold_early_stopping: Number of epochs for early stopping when model does not improve
	device: Device you want to use for training 
		--> Make sure, device exists on your hardware
After every epoch: 
	Model is saved in folder 'results'
	Results are written to console
	Results are written to 'results/results.txt'
		--> Make sure, 'results/results.txt' exists
Returns: 
  list of validation losses for every epoch
  list of training losses for every epoch
  trained model (after last epoch)
  file name for best model trained
'''
def train_classifier(model: nn.Module, 
            		     optimizer, 
            		     loss_function: nn.Module, 
            		     epochs: int = 200, 
            		     threshold_early_stopping: int = 50, 
            		     device: str = 'cpu'):
      model = model
      optimizer = optimizer
      criterion = loss_function
      val_losses = []
      losses = []
      epochs = epochs
      best_epoch = 0
      best_loss = 1000000

      model.to(device)

      for epoch in range(epochs):
        print('\n\nepoch: '+ str(epoch+1))
        model.train()
        running_loss = 0
        #previous_loss = 10000000000000
        j = 1
        for ims in iter(train_loader):
            if(j%25==0):
                   print(f'epoch: {epoch+1} \nseq: {j} of {len(train_loader)}')
            j+=1
            ims = ims.to(device)
            optimizer.zero_grad()

            bn, output = model.forward(ims)
            loss = criterion(output, ims)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()

              #if(steps%print_every ==0):
        model.eval()
        #with torch.no_grad():
        val_loss = validation(model, val_loader, criterion, device)
        losses.append(loss)
        val_losses.append(val_loss)
        #model.save_weights('Model_'+str(acc)+'_'+str(i)+'.h5')
        print('Epoch: {}/{}.. '.format(epoch+1, epochs),
            'training_loss: {:.3f}..'.format(running_loss/len(train_loader)),
            'validation loss: {:.3f}..'.format(val_loss))
        file_name = '/workspace/image_recognition_apis/Convolutional_AE/C_AE_epoch_{}_loss_{}.pth'.format(epoch, int(val_loss*100))
        torch.save(model.state_dict(), file_name)
        if(val_loss<best_loss):
            best_loss = val_loss
            best_epoch = epoch
            best_model = file_name
        elif(epoch-best_epoch > threshold_early_stopping):
            print('Training stopped')
            break
        sample_im = next(iter(test_loader))
        sample_im =  sample_im.to('cuda')
        sample_bn, sample_im = model.forward(sample_im)
        sample_im = sample_im.cpu().detach().numpy()
        sample_bn = sample_bn.cpu().detach().numpy()
        im_name = 'ims_epoch_{}.csv'.format(epoch)
        bn_name = 'bns_epoch_{}.csv'.format(epoch)
        sample_im.tofile(im_name, sep =',')
        sample_bn.tofile(bn_name, sep =',')
        running_loss = 0
        model.train()
      return(val_losses, losses, model, best_model)
'''
Define parameters for model and training here
'''
device ='cuda'
num_epochs = 200
model = AE_VGG.Autoencoder().to(torch.device(device))
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

accs, losses, model, best_model =  train_classifier(model, optimizer, criterion, num_epochs, 30, device)
