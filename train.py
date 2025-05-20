import torch.nn as nn
import torch
from torch import linalg as LA
import numpy as np
from torch.optim import Adam
import argparse
from torch.utils.data import DataLoader
from model import ExecutionLoss


#TODO write a function to train one epoch   
def train_epoch(model, optimizer, train_loader, device, epoch, log_interval=100):
    pass

#TODO write the main training function   
def train(model, optimizer, train_loader, device, epoch, log_interval=100):
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters needed for training')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    
