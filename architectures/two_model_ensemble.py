import torch
import torch.nn as nn

class TwoModelEnsembleNet(nn.Module):

    def __init__(self, model_small, model_large, freeze_models, num_classes):
        super(TwoModelEnsembleNet, self).__init__()
        self.model_small = model_small
        self.model_large = model_large

        if freeze_models:
            for param in self.model_small.parameters():
                param.requires_grad = False
            for param in self.model_large.parameters():
                param.requires_grad = False

        # get output sizes of two trained efficientnet models
        # use this if the last classifier layer is removed
        # self.joint_input_size = model_small.classifier[1].in_features + \
        #                         model_large.classifier[1].in_features

        self.joint_input_size = num_classes * 2

        self.fc = nn.Linear(self.joint_input_size, num_classes)

    def forward(self, x_small, x_large):
        model_small_output = self.model_small(x_small)
        model_large_output = self.model_large(x_large)
        out = torch.cat((model_small_output, model_large_output),1)

        out = self.fc(out)
        return out
        