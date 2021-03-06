import albumentations as A
import numpy as np
import pandas as pd
import torch
from model import prepare_model
from training import train, test
from torchvision import transforms as transforms
from dataset import Dataset

np.random.seed = 7
torch.manual_seed = 7
torch.cuda.manual_seed = 7
torch.backends.cudnn.deterministic = True

df_rle = pd.read_csv('train/train.csv')
df_imgs = pd.read_csv('HuBMAP-20-dataset_information.csv')
train_directory = './train_im/'
test_directory = './test_im/'

augmentation = A.Compose([
    A.VerticalFlip(0.5),
    A.HorizontalFlip(0.5),
    A.RandomRotate90(0.3),
    A.ShiftScaleRotate(0.2),
    A.GaussNoise(0.2)
])

transform = transforms.Compose([transforms.Resize((300, 300)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_data = Dataset(transform, augmentation, train_directory, df_rle, df_imgs)
test_data = Dataset(transform, None, test_directory, df_rle, df_imgs)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = prepare_model()
model = train(model, train_loader, test_loader, device, lr=0.01, num_epochs=20, step=100, transfer_learning=True)
