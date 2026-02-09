from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import contextlib
import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from core.config import get_model_name
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.vis import save_debug_images, save_debug_fused_images, save_debug_images_2,save_batch_fusion_heatmaps
from utils.pose_utils import align_to_pelvis

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0  # latest iter's avg
        self.avg = 0  # avg of iter 0 - now
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def collate_first_two_dims(tensor):
    dim0 = tensor.shape[0]
    dim1 = tensor.shape[1]
    left = tensor.shape[2:]
    return tensor.view(dim0 * dim1, *left)


def frozen_backbone_bn(model, backbone_name='resnet'):
    for name, m in model.named_modules():
        #print(name)
        if backbone_name in name:
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # logger.info(name)
                m.eval()
        
            else:
                m.requires_grad = False
        elif not 'warp' in name:
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
            else:
                m.requires_grad = False

@contextlib.contextmanager
def dummy_context_mgr():
    yield None


def run_model(
        config,
        dataset,
        loader,
        model,
        criterion_mse,
        criterion_mpjpe,
        final_output_dir,
        tb_writer=None,
        optimizer=None,
        epoch=None,
        is_train=True,
        **kwargs):
    # preparing meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()
    mpjpe_meters = None
    detail_mpjpes = None
    detail_preds = None
    detail_preds2d = None
    detail_weights = None
    
    nviews = len(dataset.selected_cam)
    nsamples = len(dataset) * nviews
    njoints = config.NETWORK.NUM_JOINTS
    n_used_joints = config.DATASET.NUM_USED_JOINTS
    height = int(config.NETWORK.HEATMAP_SIZE[0])
    width = int(config.NETWORK.HEATMAP_SIZE[1])
    all_view_weights = []
    all_maxvs = []
    all_nview_vis_gt = np.zeros((len(dataset), n_used_joints), dtype=np.int)


    if not is_train:
        do_save_heatmaps = kwargs['save_heatmaps']
        #all_preds = np.zeros((nsamples, njoints, 3), dtype=np.float32)
        #all_preds_3d = np.zeros((len(dataset), n_used_joints, 3), dtype=np.float32)
        if do_save_heatmaps:
            all_heatmaps = np.zeros((nsamples, njoints, height, width), dtype=np.float32)
        idx_sample = 0

    if is_train:
        phase = 'train'
        model.train()
        frozen_backbone_bn(model, backbone_name='resnet')  # do not change backbone bn params

    else:
        phase = 'test'
        model.eval()
    with dummy_context_mgr() if is_train else torch.no_grad():
        # if eval then use no_grad context manager
        end = time.time()
        for i, (input_, target_, target16_, weight_, meta_) in enumerate(loader):
            #print(i)
            #print("-------------")
            data_time.update(time.time() - end)
            debug_bit = False
            batch = input_.shape[0]

            train_2d_backbone = False
            run_view_weight = True

            input = collate_first_two_dims(input_)
            target = collate_first_two_dims(target_)
            target16 = collate_first_two_dims(target16_)
            weight = collate_first_two_dims(weight_)
            #print(is_train)
            #print(weight.shape)
            meta = dict()
            for kk in meta_:
                meta[kk] = collate_first_two_dims(meta_[kk])

            extra_params = dict()
            extra_params['run_view_weight'] = run_view_weight
            extra_params['joint_vis'] = weight
            extra_params['run_phase'] = phase
            device = torch.device('cuda:0')

            hms, extra = model(input_, target16.to(device),**meta_, **extra_params)  # todo
            
            output = hms.view(batch * nviews * nviews, njoints, *hms.shape[4:])
            origin_hms = extra['origin_hms']
            
            target_cuda = target.to(device)
            target16_cuda = target16.to(device)
            weight_cuda = weight.to(device)
            pose3d_gt = [meta_['joints_gt'][:, i, :, :].contiguous().to(device) for i in range(nviews)]
            pose3d_gt = torch.stack(pose3d_gt,dim=0)
            
            num_total_joints = batch * n_used_joints
            # --- --- forward end here

            joint_2d_loss = extra['joint_2d_loss'].mean()

            # obtain all j3d predictions
            final_preds_name = 'j3d_DenseWarper'
            pred3d = extra[final_preds_name]
            j3d_keys = []
            j2d_keys = []
            for k in extra.keys():
                if 'j3d' in k:
                    j3d_keys.append(k)
                if 'j2d' in k:
                    j2d_keys.append(k)
            # initialize only once
            if mpjpe_meters is None:
                logger.info(j3d_keys)
                mpjpe_meters = dict()
                for k in j3d_keys:
                    mpjpe_meters[k] = AverageMeter()
            if detail_mpjpes is None:
                detail_mpjpes = dict()
                for k in j3d_keys:
                    detail_mpjpes[k] = list()
            if detail_preds is None:
                detail_preds = dict()
                for k in j3d_keys:
                    detail_preds[k] = list()
                detail_preds['joints_gt'] = list()
            if detail_preds2d is None:
                detail_preds2d = dict()
                for k in j2d_keys:
                    detail_preds2d[k] = list()
            if detail_weights is None:
                detail_weights = dict()
                detail_weights['maxv'] = list()
                detail_weights['learn'] = list()

            nviews_vis = extra['nviews_vis']
            all_nview_vis_gt[i*batch:(i+1)*batch] = nviews_vis.view(batch, n_used_joints).detach().cpu().numpy().astype(np.int)
            
            joints_vis_3d = torch.as_tensor(nviews_vis >= 2, dtype=torch.float32).to(device)
            for k in j3d_keys:
                preds = extra[k] #(batch,frame,njoints,3)
                #print(preds.shape)
                if config.DATASET.TRAIN_DATASET in ['multiview_h36m','multiview_3dhp']:
                    for j in range(nviews):
                        preds[:,j,:,:] = align_to_pelvis(preds[:,j,:,:].clone().contiguous().to(device), pose3d_gt[j], 0)
                    

                selected = [meta_['joints_gt'][:, i, :, :].contiguous().clone() for i in range(nviews)]
                concatenated = torch.cat(selected, dim=1).clone()
                #print(preds[0])
                #print(concatenated.shape)
                avg_mpjpe, detail_mpjpe, n_valid_joints = criterion_mpjpe(preds.view(preds.shape[0],preds.shape[1]*preds.shape[2],*preds.shape[3:]).clone(), concatenated.to(device), joints_vis_3d=joints_vis_3d.repeat(1,4,1).clone(), output_batch_mpjpe=True)
                
                mpjpe_meters[k].update(avg_mpjpe, n=n_valid_joints)
                detail_mpjpes[k].extend(detail_mpjpe.detach().cpu().numpy().tolist())
                detail_preds[k].extend(preds.detach().cpu().numpy())
            detail_preds['joints_gt'].extend(pose3d_gt.detach().cpu().numpy())

            for k in j2d_keys:
                p2d = extra[k] #(frame,batch,nviews,3,njoints)
                p2d = p2d.permute(0, 1, 2, 4, 3).contiguous()
                p2d = p2d.detach().cpu().numpy()
                detail_preds2d[k].extend(p2d)
                
            if is_train:

                loss = torch.zeros(1,requires_grad=True).to(device)
                if train_2d_backbone:
                    hms_=hms.view(batch * nviews * nviews, njoints, *hms.shape[4:]).to(device)
                    loss =loss + criterion_mse(hms_, target16_cuda, weight_cuda.repeat(1,4,1).view(batch*nviews*nviews,njoints,1))
                loss += joint_2d_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.update(loss.item(), len(input))
            
            else:
                # validation
                loss = 0
                hms_=hms.view(batch * nviews * nviews, njoints, *hms.shape[4:])
                #print(weight.shape)
                loss_mse = criterion_mse(hms_, target16_cuda, weight_cuda.repeat(1,4,1).view(batch*nviews*nviews,njoints,1))
                loss += loss_mse
                losses.update(loss.item(), len(input))
                nimgs = input.shape[0]

            _, acc, cnt, pre = accuracy(output.detach().cpu().numpy(), target16.detach().cpu().numpy(), thr=0.083)
            
            avg_acc.update(acc, cnt)

            batch_time.update(time.time() - end)
            end = time.time()

            # ---- print logs
            #print(i)
            if i % config.PRINT_FREQ == 0 or i == len(loader)-1 or debug_bit:
                gpu_memory_usage = torch.cuda.max_memory_allocated(0)  # bytes
                gpu_memory_usage_gb = gpu_memory_usage / 1.074e9
                mpjpe_log_string = ''
                for k in mpjpe_meters:
                    mpjpe_log_string += '{:.1f}|'.format(mpjpe_meters[k].avg)
                #print(mpjpe_meters)
                msg = 'Ep:{0}[{1}/{2}]\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'Memory {memory:.2f}G\t' \
                      'MPJPEs {mpjpe_str}'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    speed=input.shape[0] / batch_time.val,
                    data_time=data_time, loss=losses, memory=gpu_memory_usage_gb, mpjpe_str=mpjpe_log_string)
                logger.info(msg)
                #break
                # ---- save debug images
                view_name = 'view_{}'.format(0)
                prefix = '{}_{}_{:08}'.format(
                    os.path.join(final_output_dir, phase), view_name, i)
                meta_for_debug_imgs = dict()
                meta_for_debug_imgs['joints_vis'] = meta['joints_vis']
                meta_for_debug_imgs['joints_2d_transformed'] = meta['joints_2d_transformed']

        # -- End epoch
        if is_train:
            pass
        else:
            cur_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
            # save mpjpes
            for k in detail_mpjpes:
                detail_mpjpe = detail_mpjpes[k]
                out_path = os.path.join(final_output_dir, '{}_ep_{}_mpjpes_{}.csv'.format(cur_time, epoch, k,))
                #np.savetxt(out_path, detail_mpjpe, delimiter=',')
                logger.info('MPJPE summary: {} {:.2f}'.format(k, np.array(detail_mpjpe).mean()))

            return 0
