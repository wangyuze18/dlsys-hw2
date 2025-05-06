# from _typeshed import Self
import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
      nn.Residual(
        nn.Sequential(
          nn.Linear(dim,hidden_dim),
          norm(hidden_dim),
          nn.ReLU(),
          nn.Dropout(p=drop_prob),
          nn.Linear(hidden_dim,dim),
          norm(dim)
        )
      ),
      nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
      nn.Linear(dim,hidden_dim),
      nn.ReLU(),
      *[ResidualBlock(
          hidden_dim, 
          hidden_dim//2,
          norm=norm,
          drop_prob=drop_prob
        ) for _ in range(num_blocks)],
      nn.Linear(hidden_dim,num_classes)
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    total_acc, total = 0, len(dataloader.dataset)
    loss_func = nn.SoftmaxLoss()
    total_loss = 0.0
    if opt is not None:
        model.train()
        for idx, data in enumerate(dataloader):
            x, y = data
            output = model(x)
            opt.reset_grad()
            loss = loss_func(output, y)
            total_loss += loss.numpy() * x.shape[0]
            loss.backward()
            opt.step()
            total_acc += (y.numpy() == output.numpy().argmax(1)).sum()
    else:
        model.eval()
        for idx, data in enumerate(dataloader):
            x, y = data
            output = model(x)
            loss = loss_func(output, y)
            total_loss += loss.numpy() * x.shape[0]
            total_acc += (y.numpy() == output.numpy().argmax(1)).sum()
    error_rate = (total - total_acc) / total
    return error_rate, total_loss / total    
    ### END YOUR SOLUTION

def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(os.path.join(data_dir,'train-images-idx3-ubyte.gz'),
                      os.path.join(data_dir,'train-labels-idx1-ubyte.gz'))
    test_dataset = ndl.data.MNISTDataset(os.path.join(data_dir,'t10k-images-idx3-ubyte.gz'),
                      os.path.join(data_dir,'t10k-labels-idx1-ubyte.gz'))
    train_dataloader = ndl.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset,batch_size=batch_size)
    model = MLPResNet(784,hidden_dim)
    opt = optimizer(model.parameters(),lr=lr,weight_decay=weight_decay) if optimizer is not None else None
    for i in range(epochs):
        train_error, train_loss = epoch(train_dataloader, model, opt)
        # test_error, test_loss = epoch(test_dataloader, model)
        # print(f"Epoch {i+1}/{epochs} Train Error: {train_error:.4f} Train Loss: {train_loss:.4f} Test Error: {test_error:.4f} Test Loss: {test_loss:.4f}")
    test_error, test_loss = epoch(test_dataloader, model)
    return (train_error, train_loss, test_error, test_loss)
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
