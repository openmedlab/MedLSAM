#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
#os._exit(00)
import sys
sys.path.append(os.path.abspath(__file__)) 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
from networks.medlam import MedLAM

class NetFactory(object):

    @staticmethod
    def create(name):
        if name == 'MedLAM':
            return MedLAM
        
        # add your own networks here
        print('unsupported network:', name)
        exit()
