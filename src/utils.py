from typing import Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim

def TrainNet(net: nn.Module, 
              X: torch.tensor,
              Y: torch.tensor, 
              optimizer: optim.Optimizer,
              target_loss: Optional[float] = 1e-4,
              max_epochs: Optional[int] = 1000):
    
    epochs = 0
    loss = torch.tensor(1.0)
    loss_fn = nn.MSELoss(reduction='mean')
    while (loss.item() > target_loss and epochs<max_epochs):
        optimizer.zero_grad()
        output = net(X)
        loss = loss_fn(output, Y)
        loss.backward()
        optimizer.step()
        epochs += 1
    print("Final Loss: {}".format(loss.item()))
    return net

def loss(X: torch.tensor,
         Y: torch.tensor,
         weights: torch.tensor):
    output = torch.matmul(X, weights)
    loss = torch.norm(output-Y)**2
    n = X.size()[0]
    loss = loss/n
    return loss.item()


