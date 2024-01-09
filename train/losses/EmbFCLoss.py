import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


class EmbFCLoss(nn.Module):
    def __init__(self, dis_ratio, rdrloss, embratio=10, crop_pad=0, point_num=20, dis_thresh=1.5):
        super(EmbFCLoss, self).__init__()
        self.dis_ratio = dis_ratio
        self.rdrloss = rdrloss
        self.embratio = embratio
        self.crop_pad = crop_pad
        self.point_num = point_num
        self.dis_thresh = dis_thresh
    def random_position(self, shape):
        position = []
        for i in range(len(shape)):
            position.append(random.randint(self.crop_pad[i], shape[i]- self.crop_pad[i]))
        return position
    
    def forward(self, ori_img_batch0, ori_img_batch1, fs_po0, fs_po1, ori_fs_po0, ori_fs_po1,
                    fc_label_batch, ori_fc_label_batch, predic_0, 
                    predic_1, ori_predic_0, ori_predic_1):
        
        fc_0, fc_1 =  predic_0['fc_position'], predic_1['fc_position']
        ori_fc_0, ori_fc_1 = ori_predic_0['fc_position'], ori_predic_1['fc_position']
        shape = fs_po0.shape[2::] # d w h
        
        #fc_predic = self.dis_ratio.cuda()*torch.tanh(fc_0-fc_1)
        #fc_loss = self.rdrloss(fc_predic, fc_label_batch)
        ori_fc_predic = self.dis_ratio.cuda()*torch.tanh(ori_fc_0-ori_fc_1)
        ori_fc_loss = self.rdrloss(ori_fc_predic, ori_fc_label_batch)
        emb_loss = torch.tensor(0.0).cuda()
        n = 0
        feature0 = {}
        feature1 = {}
        ori_feature_vec0={}
        ori_feature_vec1={}
        for i in range(3): # The network outputs feature maps at three levels
            cfeature0 = predic_0['feature{0:}'.format(i)]
            feature0[i] = F.normalize(cfeature0.reshape(cfeature0.shape[0], 1, cfeature0.shape[1], -1), dim=2) # bs * 1 * c * dwh

            cfeature1 = predic_1['feature{0:}'.format(i)]
            feature1[i] = F.normalize(cfeature1.reshape(cfeature1.shape[0], 1, cfeature1.shape[1], -1), dim=2) # bs * 1 * c * dwh
            
        
        for ii in range(self.point_num): #Twenty points were randomly selected as feature vectors
            bg_choose = True
            bg_count = 0
            while bg_choose:
                bg_count += 1    
                random_cor0 = [random.randint(0, shape[iii]-1) for iii in range(3)]
                if torch.sum(ori_img_batch0[:,:,random_cor0[0],random_cor0[1],random_cor0[2]]>=0.0001)>ori_img_batch0.shape[0]//2 or bg_count >= 20: # At least half of the selected sites were non-background
                    bg_choose = False
                    
            bg_choose = True
            while bg_choose:
                random_cor1 = [random.randint(0, shape[iii]-1) for iii in range(3)]
                bg_count += 1     
                if torch.sum(ori_img_batch1[:,:,random_cor1[0],random_cor1[1],random_cor1[2]]>=0.0001)>ori_img_batch1.shape[0]//2 or bg_count >= 20: # At least half of the selected sites were non-background
                    bg_choose = False       
            ori_select_cor0 = ori_fs_po0[:, :, random_cor0[0]:random_cor0[0]+1,
                                    random_cor0[1]: random_cor0[1]+1, random_cor0[2]:random_cor0[2]+1]# bs, 3, 1, 1, 1
            ori_select_cor1 = ori_fs_po1[:, :, random_cor1[0]:random_cor1[0]+1,
                                    random_cor1[1]: random_cor1[1]+1, random_cor1[2]:random_cor1[2]+1] # bs, 3, 1, 1, 1
            if n == 0:
                dis2select_cor0 = torch.norm(fs_po0-ori_select_cor0, dim=1).unsqueeze(1) # bs,1,d,w,h ; The position of the original point on the map after deformation
                dis2select_cor1 = torch.norm(fs_po1-ori_select_cor1, dim=1).unsqueeze(1)  # bs,1,d,w,h
            else:
                dis2select_cor0 = torch.cat((dis2select_cor0, torch.norm(fs_po0-ori_select_cor0, dim=1).unsqueeze(1)), dim=1) # bs,n,d,w,h
                dis2select_cor1 = torch.cat((dis2select_cor1, torch.norm(fs_po1-ori_select_cor1, dim=1).unsqueeze(1)), dim=1)  # bs,n,d,w,h


            for iiii in range(3):
                downscale = 2**(2-iiii)
                cori_feature_vec0 = ori_predic_0['feature{0:}'.format(iiii)] \
                                [:, :, np.trunc(random_cor0[0]/downscale).astype(np.int8), \
                                np.trunc(random_cor0[1]/downscale).astype(np.int8), \
                                np.trunc(random_cor0[2]/downscale).astype(np.int8)] # choosen feature vector
                cori_feature_vec0 = F.normalize(cori_feature_vec0, dim=1).unsqueeze(dim=1) # bs *1 *c
                cori_feature_vec1 = ori_predic_1['feature{0:}'.format(iiii)]\
                                [:, :, np.trunc(random_cor1[0]/downscale).astype(np.int8), \
                                np.trunc(random_cor1[1]/downscale).astype(np.int8), \
                                np.trunc(random_cor1[2]/downscale).astype(np.int8)]
                cori_feature_vec1 = F.normalize(cori_feature_vec1, dim=1).unsqueeze(dim=1)
                if n == 0:
                    ori_feature_vec0[iiii] = cori_feature_vec0
                    ori_feature_vec1[iiii] = cori_feature_vec1
                else:
                    ori_feature_vec0[iiii]=torch.cat((ori_feature_vec0[iiii], cori_feature_vec0), dim=1) # bs* n* c
                    ori_feature_vec1[iiii]=torch.cat((ori_feature_vec1[iiii], cori_feature_vec1), dim=1)
            n +=1
        for iiii in range(3):
            cur_dis0 = F.interpolate(dis2select_cor0,scale_factor=1/2**(2-iiii), mode='trilinear') # bs,27,d,w,h
            cur_dis1 = F.interpolate(dis2select_cor1,scale_factor=1/2**(2-iiii), mode='trilinear')
            cur_dis0 = cur_dis0.reshape(cur_dis0.shape[0], cur_dis0.shape[1],-1)
            cur_dis1 = cur_dis1.reshape(cur_dis1.shape[0], cur_dis1.shape[1],-1)
            prob0 = torch.softmax(15*torch.sum(feature0[iiii]* \
                    ori_feature_vec0[iiii].unsqueeze(dim=-1), dim=2), dim=2) # bs*n *dwh
            prob1 = torch.softmax(15*torch.sum(feature1[iiii]* \
                    ori_feature_vec1[iiii].unsqueeze(dim=-1), dim=2), dim=2)
                                  
            pos_mask0 = (1*(cur_dis0 <= 2**(2-iiii)*self.dis_thresh)).to(torch.float32)
            neg_mask0 = (1*(cur_dis0 >= 2**(2-iiii)*2*self.dis_thresh)).to(torch.float32)
            pos_mask1 = (1*(cur_dis1 <= 2**(2-iiii)*self.dis_thresh)).to(torch.float32)
            neg_mask1 = (1*(cur_dis1 >= 2**(2-iiii)*2*self.dis_thresh)).to(torch.float32)
            # emb_loss += -torch.sum(pos_mask0*torch.log(prob0))-torch.mean(neg_mask0*torch.log(1-prob0), dim=2).sum()\
            #     -torch.sum(pos_mask1*torch.log(prob1))-torch.mean(neg_mask1*torch.log(1-prob1), dim=2).sum()
            emb_loss += -torch.sum(pos_mask0*torch.log(prob0))/torch.sum(pos_mask0)-torch.sum(neg_mask0*torch.log(1-prob0))/torch.sum(neg_mask0)\
                -torch.sum(pos_mask1*torch.log(prob1))/torch.sum(pos_mask1)-torch.sum(neg_mask1*torch.log(1-prob1))/torch.sum(neg_mask1) 
        emb_loss /= (3*self.point_num)             
        train_loss = ori_fc_loss+emb_loss
        return [train_loss, ori_fc_loss, emb_loss, ori_fc_predic]