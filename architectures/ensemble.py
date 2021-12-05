import torch
import torch.nn as nn

class FCEnsembleNet(nn.Module):

    def __init__(self, models, joint_input_size, num_classes):
        super(FCEnsembleNet, self).__init__()
        self.models = nn.ModuleList(models)

        self.fc = nn.Linear(joint_input_size, num_classes)

    def forward(self, x):
        out = None
        for model in self.models:
            model_output = model(x)
            out = torch.cat((model_output, out),1)

        out = self.fc(out)
        return out