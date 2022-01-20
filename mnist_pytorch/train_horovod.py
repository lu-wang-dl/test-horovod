# Databricks notebook source
# MAGIC %md # Model runner - Horovod cluster distributed training

# COMMAND ----------

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import horovod.torch as hvd
from sparkdl import HorovodRunner
import os

# COMMAND ----------

import torch
import torch.nn as nn
import torch.nn.functional as F
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
 
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# COMMAND ----------

def train_one_epoch(model, device, data_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader) * len(data),
                100. * batch_idx / len(data_loader), loss.item()))

# COMMAND ----------

# Specify training parameters
batch_size = 100
num_epochs = 3
momentum = 0.5
log_interval = 100

# COMMAND ----------

def save_checkpoint(log_dir, model, optimizer, epoch):
  filepath = log_dir + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
  state = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
  }
  torch.save(state, filepath)
  
def load_checkpoint(log_dir, epoch=num_epochs):
  filepath = log_dir + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
  return torch.load(filepath)
 
def create_log_dir(logging_dir):
  # Check whether the specified path exists or not
  isExist = os.path.exists(logging_dir)
  if not isExist:
    # Create a new directory because it does not exist 
    os.makedirs(logging_dir)
  return logging_dir

# COMMAND ----------

# MAGIC %md Parameters

# COMMAND ----------

dbutils.widgets.text("batch_size", '100')
batch_size = int(dbutils.widgets.get("batch_size"))

dbutils.widgets.text("num_epochs", '3')
num_epochs = int(dbutils.widgets.get("num_epochs"))

dbutils.widgets.text("momentum", '0.5')
momentum = float(dbutils.widgets.get("momentum"))

dbutils.widgets.text("log_interval", '100')
log_interval = int(dbutils.widgets.get("log_interval"))

dbutils.widgets.text("logging_dir", '/dbfs/ml/horovod_pytorch')
logging_dir = dbutils.widgets.get("logging_dir")

# COMMAND ----------

# MAGIC %md  Run distributed Horovod workflow

# COMMAND ----------

hvd_log_dir = create_log_dir(logging_dir)
print("Log directory:", hvd_log_dir)

def train_hvd(learning_rate):
  
  # Initialize Horovod
  hvd.init()  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  if device.type == 'cuda':
    # Pin GPU to local rank
    torch.cuda.set_device(hvd.local_rank())

  train_dataset = datasets.MNIST(
    # Use different root directory for each worker to avoid conflicts
    root='/data-%d'% hvd.rank(),  
    train=True, 
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
  )

  from torch.utils.data.distributed import DistributedSampler
  
  # Configure the sampler so that each worker gets a distinct sample of the input dataset
  train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
  # Use train_sampler to load a different sample of data on each worker
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

  model = Net().to(device)
  
  # The effective batch size in synchronous distributed training is scaled by the number of workers
  # Increase learning_rate to compensate for the increased batch size
  optimizer = optim.SGD(model.parameters(), lr=learning_rate * hvd.size(), momentum=momentum)

  # Wrap the local optimizer with hvd.DistributedOptimizer so that Horovod handles the distributed optimization
  optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
  
  # Broadcast initial parameters so all workers start with the same parameters
  hvd.broadcast_parameters(model.state_dict(), root_rank=0)

  for epoch in range(1, num_epochs + 1):
    train_one_epoch(model, device, train_loader, optimizer, epoch, log_interval)
    # Save checkpoints only on worker 0 to prevent conflicts between workers
    if hvd.rank() == 0:
      save_checkpoint(hvd_log_dir, model, optimizer, epoch)

# COMMAND ----------

# MAGIC %md Train

# COMMAND ----------

hr = HorovodRunner(np=2) 
hr.run(train_hvd, learning_rate = 0.001)

# COMMAND ----------

# MAGIC %md Test

# COMMAND ----------

test(hvd_log_dir, num_epochs)

# COMMAND ----------


