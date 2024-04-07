from torch import nn

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FNN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

        self.act = nn.ReLU()
        self.output = nn.Sigmoid()
        

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.output(x)

        return x.squeeze(-1)