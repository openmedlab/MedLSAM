import torch
import shutil
import os

class Save_checkpoint(object):
    def __init__(self):
        self.pre_check_save_name = ''
        self.pre_best_save_name = ''
        self.delete_pre = False

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar', bestname='model_best.pth.tar'):
        torch.save(state, filename)
        print('succeffcully save', filename)
        if self.delete_pre:
            if os.path.exists(self.pre_check_save_name):
                os.remove(self.pre_check_save_name)
        self.pre_check_save_name = filename
        if is_best:
            if self.delete_pre:
                if os.path.exists(self.pre_best_save_name):
                    os.remove(self.pre_best_save_name)
            self.pre_best_save_name = bestname
            shutil.copyfile(filename, bestname)
        self.delete_pre = True

def load_weights(config_train, net, optimizer):
    if os.path.isfile(config_train['pretrained_model_path']):
        print("=> loading checkpoint '{}'".format(config_train['pretrained_model_path']))
        if config_train['pretrained_model_path'].endswith('.tar'):
            checkpoint = torch.load(config_train['pretrained_model_path'])
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        elif config_train['pretrained_model_path'].endswith('.pkl'):
            net_weight = torch.load(config_train['pretrained_model_path'], map_location=lambda storage, loc: storage)
            net.load_state_dict(net_weight)
        print("=> loaded checkpoint '{}' ".format(config_train['pretrained_model_path']))
    else:
        raise(ValueError("=> no checkpoint found at '{}'".format(config_train['pretrained_model_path'])))
    return net, optimizer