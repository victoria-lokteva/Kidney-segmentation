import torch
import torch.nn as nn
from tqdm.notebook import tqdm as tqdm
from dataset import Dataset


def create_loaders(transform, train_directory, test_directory,df_rle, df_imgs, batch_size):
    train_data = Dataset(transform, train_directory, df_rle, df_imgs)
    test_data = Dataset(transform, test_directory, df_rle, df_imgs)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size)
    return train_loader, test_loader


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, pred, target, smooth=0.1):
        # smooth helps avoid devision by zero
        a = torch.sum(pred)
        b = torch.sum(target)
        intersection = torch.sum(pred * target)
        dice = 2 * (intersection + smooth) / (a + b + smooth)
        softdice = 1 - dice
        return softdice


def test(net, test_loader, device):

    loss_f = SoftDiceLoss()
    total_loss = 0
    test_loader = iter(test_loader)

    net = net.eval()
    with torch.no_grad():
        for idx, (image, target) in enumerate(test_loader):
            image = image.to(device)
            target = target.to(device)
            pred = net(image)
            loss = loss_f(pred, target)
            total_loss += loss
        print(total_loss)
    return pred


def train(net, train_loader, test_loader, device, lr=0.01, num_epochs=30, step=100, transfer_learning=False):

    net = net.to(device)
    loss_f = SoftDiceLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step)

    for epoch in tqdm(range(num_epochs)):
        train_loader = iter(train_loader)

        for idx, (image, target) in tqdm(enumerate(train_loader)):
            image = image.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            if transfer_learning:
                output = net(image)
                output = output['out']
            else:
                output = net(image)
            loss = loss_f(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

        test(net, test_loader, device)

    return net



