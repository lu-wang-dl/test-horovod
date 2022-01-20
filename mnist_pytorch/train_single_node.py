# Databricks notebook source
# MAGIC %md # Model runner - single node training

# COMMAND ----------

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

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

def train_one_epoch(model, device, data_loader, optimizer, epoch):
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

def test(log_dir):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  loaded_model = Net().to(device)
  
  checkpoint = load_checkpoint(log_dir)
  loaded_model.load_state_dict(checkpoint['model'])
  loaded_model.eval()
 
  test_dataset = datasets.MNIST(
    'data', 
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
  data_loader = torch.utils.data.DataLoader(test_dataset)
 
  test_loss = 0
  for data, target in data_loader:
      data, target = data.to(device), target.to(device)
      output = loaded_model(data)
      test_loss += F.nll_loss(output, target)
  
  test_loss /= len(data_loader.dataset)
  print("Average test loss: {}".format(test_loss.item()))

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
 
def create_log_dir():
  log_dir = os.path.join(PYTORCH_DIR, str(time()), 'MNISTDemo')
  os.makedirs(log_dir)
  return log_dir

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

# MAGIC %md  Run single-node workflow

# COMMAND ----------

single_node_log_dir = create_log_dir(logging_dir)
print("Log directory:", single_node_log_dir)


def train(learning_rate):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  train_dataset = datasets.MNIST(
    '/data', 
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
  data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  model = Net().to(device)

  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

  for epoch in range(1, num_epochs + 1):
    train_one_epoch(model, device, data_loader, optimizer, epoch, log_interval)
    save_checkpoint(single_node_log_dir, model, optimizer, epoch)

# COMMAND ----------

# MAGIC %md Train

# COMMAND ----------

train(learning_rate = 0.001)

# COMMAND ----------

# MAGIC %md Test

# COMMAND ----------

test(single_node_log_dir, num_epochs)
