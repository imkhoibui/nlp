from torch import nn

class ClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes

        

    def forward(self):
        return