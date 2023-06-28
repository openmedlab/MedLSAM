from __future__ import print_function
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from networks.layers import SingleConv3D
import math
import numpy as np


class MedLAM(nn.Module):
    def __init__(self,  inc=1, patch_size=1, n_classes=5, base_chns=12, droprate=0, norm='in', depth = False, dilation=1):
        super(MedLAM, self).__init__()
        self.model_name = "seg"
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')  # 1/4(h,h)
        self.downsample = nn.MaxPool3d(2, 2)  # 1/2(h,h)
        self.drop = nn.Dropout(droprate)

        self.conv0_1 = SingleConv3D(inc, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv0_2 = SingleConv3D(2*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv1_1 = SingleConv3D(2*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv1_2 = SingleConv3D(2*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv2_1 = SingleConv3D(2*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv2_2 = SingleConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv3_1 = SingleConv3D(4*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv3_2 = SingleConv3D(4 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv4_1 = SingleConv3D(8*base_chns, 8*base_chns, norm=norm, depth=depth, dilat=math.ceil(dilation/2), pad='same')
        self.conv4_2 = SingleConv3D(8 * base_chns, 16 * base_chns, norm=norm, depth=depth, dilat=math.ceil(dilation/2), pad='same')

        self.conv5_1 = SingleConv3D(24*base_chns, 8*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv5_2 = SingleConv3D(8 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv6_1 = SingleConv3D(12*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv6_2 = SingleConv3D(4 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv7_1 = SingleConv3D(6*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv7_2 = SingleConv3D(4* base_chns,  4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv8_1 = SingleConv3D(6*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv8_2 = SingleConv3D( 4*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.classification = nn.Sequential(
            nn.Conv3d(in_channels=4*base_chns, out_channels=n_classes, kernel_size=1),
        )
        fc_inc = int(np.asarray(patch_size).prod()/4096)*16*base_chns
        self.fc1 = nn.Linear(fc_inc, 8 * base_chns)
        self.fc2 = nn.Linear(8 * base_chns, 4 * base_chns)
        self.fc3 = nn.Linear(4 * base_chns, 3)

    def forward(self, x, out_fc=False, decoder=True, out_feature=False, out_classification=True, feature_norm=False):
        result = {}
        out = self.conv0_1(x)
        conv0 = self.conv0_2(out)
        out = self.downsample(conv0)
        out = self.conv1_1(out)
        conv1 = self.conv1_2(out)
        out = self.downsample(conv1)  # 1/2
        out = self.conv2_1(out)
        conv2 = self.conv2_2(out)  #
        out = self.downsample(conv2)  # 1/4
        out = self.conv3_1(out)
        conv3 = self.conv3_2(out)  #
        out = self.downsample(conv3)  # 1/8
        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.drop(out)

        if out_fc:
            fc_out = out.view(out.shape[0],-1)
            fc_out = self.fc1(fc_out)
            fc_out = self.fc2(fc_out)
            fc_out = self.fc3(fc_out)
            result['fc_position'] = fc_out

        if decoder:
            up5 = self.upsample(out)  # 1/4
            out = t.cat((up5, conv3), 1)
            out = self.conv5_1(out)
            out = self.conv5_2(out)

            out = self.upsample(out)  # 1/2
            out = t.cat((out, conv2), 1)
            out = self.conv6_1(out)
            feature0 = self.conv6_2(out)

            out = self.upsample(feature0)
            out = t.cat((out, conv1), 1)
            out = self.conv7_1(out)
            feature1 = self.conv7_2(out)

            out = self.upsample(feature1)
            out = t.cat((out, conv0), 1)
            out = self.conv8_1(out)
            feature2 = self.conv8_2(out)
            if out_feature:
                if feature_norm:
                    result['feature0'] = F.normalize(feature0, dim=1)
                    result['feature1'] = F.normalize(feature1, dim=1)
                    result['feature2'] = F.normalize(feature2, dim=1)
                else:
                    result['feature0'] = feature0
                    result['feature1'] = feature1
                    result['feature2'] = feature2
            
            if out_classification:
                out = self.classification(feature2)
                result['prob'] = t.sigmoid(out)
        
        return result
