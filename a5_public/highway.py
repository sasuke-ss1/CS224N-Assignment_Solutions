#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, e_word):
        super().__init__()
        self.e_word = e_word
        self.proj = e_word
        self.linear = nn.Linear(e_word, e_word)
        self.gate = nn.Linear(e_word,e_word)
        
    def forward(self, x_conv_out):
        x_porj = torch.relu(self.linear(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))
        x_highway = torch.mul(x_porj, x_gate) + torch.mul((1- x_gate), x_conv_out)
        return(x_highway)
        

### END YOUR CODE 

