#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    def __init__(self, n_embed, num_filter, k=5, m_word = 21):
        super().__init__()
        self.input = n_embed
        self.output = num_filter
        self.conv1 = nn.Conv1d(self.input, self.output, kernel_size=k)
        self.maxpool = nn.MaxPool1d(kernel_size=m_word - k + 1)
        
    def forward(self, x_reshaped):
        x_conv = self.maxpool(torch.relu(self.conv1(x_reshaped)))
        
        return torch.squeeze(x_conv, -1)
        

### END YOUR CODE

