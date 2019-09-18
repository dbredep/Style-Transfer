import torch
import torch.nn as nn

class GradManip(nn.Module):
    def __init__(self, flip_rate=-1):
        super(GradManip, self).__init__()
        self.flip_rate = flip_rate

    def forward(self, x):
        return x

    def backward(self, grad_output):
        #print('Grad_Output')
        #print(grad_output.shape)
        return self.flip_rate * grad_output