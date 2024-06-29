import torch


class Softmax():
    def __init__(self, data):
        '''
        Initialize the softmax class
        '''
        self.data = data

    def softmax(self, x):
        '''
        Calculate the softmax of a vector x 
        '''
        result = torch.exp(x) / torch.sum(torch.exp(x))
        return result

    def softmax_table(self, x):
        '''
        Calculate the softmax table of a matrix x
        '''
        c = torch.max(x)
        result = torch.exp(x-c) / torch.sum(torch.exp(x-c))
        return result
