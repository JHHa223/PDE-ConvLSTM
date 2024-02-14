import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DataParallel
import torch.backends.cudnn as cudnn
import numpy as np
from loader import Loader
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device: ', device)
print('Current cuda device: ', torch.cuda.current_device())
print('Count of using GPUs: ', torch.cuda.device_count())

# Create the model
model = Model_ConvLSTM(nf=12,in_chan=1)
model = model.to(device)
model = torch.nn.DataParallel(model)
cudnn.benchmark = True

# Define loss and optimization functions
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.MSELoss()
#
trainset = Loader(is_train=0)
testset = Loader(is_train=1)
#
train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4, shuffle=True,
    num_workers=40, pin_memory=True)
#
test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=4, shuffle=True,
    num_workers=40, pin_memory=True)
#
#
print("DATA READING DONE")
# Training loop
def train(epoch):
    model.train()
    for batch_idx, inputs in enumerate(train_loader):
        inputs, targets = inputs[:,:12,:,:,:].float().permute(0,1,4,3,2).to(device), inputs[:,12:,:,:,:].float().permute(0,1,4,3,2).to(device)
            
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        print(loss)
        loss.backward()
        optimizer.step()

    scheduler.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item()}')
    # Save model checkpoint at the end of each epoch
    state = model.state_dict()
    torch.save(state, 'checkpoints/ckpt_Epoch'+str(epoch+1)+'.pth')

def test(epoch):
    model.eval()
    with torch.no_grad():
        for batch_idx, inputs in enumerate(test_loader):
            inputs, targets = inputs[:,:12,:,:,:].float().permute(0,1,4,3,2).to(device), inputs[:,12:,:,:,:].float().permute(0,1,4,3,2).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {loss.item()}')

num_epochs = 40
for epoch in range(num_epochs):
    print("TRAINING")
    train(epoch)
    test(epoch)
