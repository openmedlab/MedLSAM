import os
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path', type=str, default='train/checkpoint/your.tar')
    config_file = parser.parse_args().config

    checkpoint = torch.load(config_file)
    state_dict = checkpoint['state_dict']
    
    torch.save(state_dict, 'checkpoint/medlam.pth')