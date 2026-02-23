from time import sleep
import torch
import math
from torch import nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
    def forward(self, x):
        return self.layers(x)

def main():
    dev = 'cpu'
    test_model = TestModel().to(dev)

    X = torch.rand(100, 10)
    y = X.sum(dim =1, keepdim=True)

    mse_function = nn.MSELoss()
    optim = torch.optim.Adam(test_model.parameters(), lr=0.01)

    for epoch in range(100):
        prediction = test_model(X)
        loss = mse_function(prediction, y)

        optim.zero_grad()
        loss.backward()
        optim.step()
        if epoch % 20 == 0:
            print(f'Epoch: {epoch} & loss: {loss.item()}')
            sleep(0.5)
    
    final_prediction = test_model(torch.tensor([1.0]*10)).item()
    print(f'NN prediction is: {final_prediction}\n Actual goal: 10')

if __name__ == "__main__":
    main()