import torch
import torch.nn as nn
from models.epipolar_fusion_layer import CamFusionModule, get_inv_cam, get_inv_affine_transform

class PoseEstimationModel(nn.Module):
    def __init__(self, nview, njoints, h, w, joint_hm_mapping, config):
        super(PoseEstimationModel, self).__init__()
        self.nview = nview
        self.njoints = njoints
        self.h = h
        self.w = w
        self.epipolar_layer=CamFusionModule(nview,njoints,h,w,joint_hm_mapping,config)
    def forward(self, heatmaps ,affine_trans, cam_Intri, cam_R, cam_T, inv_affine_trans):
        batch_size = heatmaps.size(0) // self.nview
        enhance_x=self.epipolar_layer(heatmaps, affine_trans, cam_Intri, cam_R, cam_T, inv_affine_trans)
        device=heatmaps.device
        output=torch.ones((batch_size,self.nview,self.nview,self.njoints,self.h,self.w)).to(device)
        for batch in range(batch_size):
            for curview in range(self.nview):
                for othview in range(self.nview):
                    if curview==othview:
                        output[batch][curview][othview]=heatmaps[batch*self.nview+curview]
                    else:
                        othview_=othview
                        if curview<othview:
                            othview_=othview_-1
                        output[batch][curview][othview]=0.5*enhance_x[batch*self.nview+curview][othview_]+0.5*heatmaps[batch*self.nview+curview]
        return output