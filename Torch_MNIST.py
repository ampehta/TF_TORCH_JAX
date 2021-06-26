import torch 
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class mTorch(nn.Module):
    def __init__(self):
        super(mTorch,self).__init__()
        self.dense1 = nn.Linear(784,128)
        self.output = nn.Linear(128,10)

    def forward(self,x):
        x = x.view(-1,784)
        x = F.relu(self.dense1(x))

        return self.output(x)

if __name__ == '__main__':
  download_root = './MNIST'
  train_ds = MNIST(download_root,train=True,transform = transforms.ToTensor(), download=True)
  test_ds  = MNIST(download_root,train=False,transform = transforms.ToTensor(), download=True)

  train_dataloader = DataLoader(train_ds,batch_size = 32, drop_last = True) # drop_last 마지막 배치가 32개가 안되는 경우 제외
  
  if torch.cuda.is_available():
    device = torch.device('cuda')
    print(device)
    
  mTorch = mTorch().to(device)
  loss_obj = nn.CrossEntropyLoss().to(device)
  optimizer = torch.optim.Adam(mTorch.parameters(),lr=0.001)
  
  epochs = 6 
  
  for epoch in range(epochs):
    loss_ = []
    accuracy = []

    for data,target in train_dataloader:

        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        pred = mTorch(data)
        loss = loss_obj(pred,target)

        loss.backward()
        optimizer.step()

        loss_.append(loss)
        accuracy.append((torch.argmax(pred,1)==target).sum()/32)

    print(f'Epoch: {epoch+1} | Loss: {sum(loss_)/len(train_dataloader)} | Accuracy: {sum(accuracy)/len(train_dataloader)}')
