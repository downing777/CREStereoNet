# from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from nets import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
from matplotlib import pyplot as plt
# from apex import amp
import cv2

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
#设置基本路径与训练参数

parser = argparse.ArgumentParser(description='CREStereo_Net')
parser.add_argument('--model', default='crestereo', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="/datasets/sceneflow/", help='data path')
parser.add_argument('--trainlist', default='./filenames/sceneflow_trainsmall.txt', help='training list')
parser.add_argument('--testlist',default='./filenames/sceneflow_testsmall.txt', help='testing list')
parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=2, help='testing batch size')
parser.add_argument('--epochs', type=int, default=32, help='number of epochs to train')
parser.add_argument('--lrepochs',default="20,32,40,48,56:2", type=str,  help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir',default='.', help='the directory to save logs and checkpoints')
#parser.add_argument('--loadckpt', default='./pretrained_model/train2.1.ckpt',help='load the weights from a specific checkpoint')

parser.add_argument('--loadckpt', default=False,help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=16, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=False)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=8, drop_last=False)

# model, optimizer
gpus = [0, 1]
model = __models__[args.model](args.maxdisp, False, False)
model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
#optimizer = nn.DataParallel(optimizer, device_ids=gpus)
#optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=0.9)

# load parameters
start_epoch = 0
#继续训练
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
#加载预训练模型
elif args.loadckpt:
    #load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    model_dict.update(pre_dict) 
    model.load_state_dict(model_dict)
print("start at epoch {}".format(start_epoch))

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8):
    '''
    valid: (2, 384, 512) (B, H, W) -> (B, 1, H, W)
    flow_preds[0]: (B, 2, H, W)
    flow_gt: (B, 2, H, W)
    '''
    n_predictions = len(flow_preds)
    flow_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = torch.abs(flow_preds[i] - flow_gt)
        flow_loss += i_weight * (valid.unsqueeze(1) * i_loss).mean()

    return flow_loss

def train():
    torch.cuda.empty_cache()
    loss_c=[]
    epoch_c=[]
    test_c=[]
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)
        epoch_c.append(epoch_idx)
        # training
        loss_a=0
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            loss_a+=loss/23
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                     len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))
        loss_c.append(loss_a)  
        # saving checkpoints

        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            #id_epoch = (epoch_idx + 1) % 100
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()
        '''
        if (epoch_idx) % 1 == 0:
            with torch.no_grad():
        # # testing
                torch.cuda.empty_cache()
                avg_test_scalars = AverageMeterDict()
                test_a=0
                for batch_idx, sample in enumerate(TestImgLoader):
                    global_step = len(TestImgLoader) * epoch_idx + batch_idx
                    start_time = time.time()
                    do_summary = global_step % args.summary_freq == 0
                    loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
                    test_a+=loss/23
                    if do_summary:
                        save_scalars(logger, 'test', scalar_outputs, global_step)
                        save_images(logger, 'test', image_outputs, global_step)
                    avg_test_scalars.update(scalar_outputs)
                    del scalar_outputs, image_outputs
                    print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                         batch_idx,
                                                                                         len(TestImgLoader), loss,
                                                                                        time.time() - start_time))
                test_c.append(test_a)
                avg_test_scalars = avg_test_scalars.mean()
                save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
                print("avg_test_scalars", avg_test_scalars)
                gc.collect()
        '''


    plt.plot(epoch_c, loss_c)
    #plt.plot(epoch_c, test_c)
    plt.show()

# train one sample

def train_sample(sample, compute_metrics=False):
    model.train()
    imgL, imgR, disp_gt= sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda() # (B, H, W)
    optimizer.zero_grad()
    flow_predictions = model(imgL, imgR) # flow_preds[0]: (B, 2, H, W)  (B, H, W)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)

    gt_disp = torch.unsqueeze(disp_gt, dim=1)  # [2, 384, 512] -> [2, 1, 384, 512]
    gt_flow = torch.cat([gt_disp, gt_disp * 0], dim=1)  # [2, 2, 384, 512]

    loss = sequence_loss(
                flow_predictions, gt_flow, mask, gamma=0.8
            ) # (3, B, H, W)  flow_predictions (3, B, 2, H, W)

    disp_ests = [disp_est[:, 0, :, :] for disp_est in flow_predictions] # (3, B, H, W)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    loss.backward()
    optimizer.step()
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# test one sample
def inference(imgL, imgR, model, n_iter=20):

	print("Model Forwarding...")
	imgL_dw2 = F.interpolate(
		imgL,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	imgR_dw2 = F.interpolate(
		imgR,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	# print(imgR_dw2.shape)
	with torch.inference_mode():
		pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)

		pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
	pred_disp = pred_flow[:, 0, :, :]

	return pred_disp

@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()
    imgL, imgR, disp_gt, wid, hit = sample['left'], sample['right'], sample['disparity'],sample['wid'],sample['hit']
    wid=int(wid[0])
    hit=int(hit[0])
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)


    #disp_ests = disp_ests[torch.arange(disp_ests.size(0))!=0]    
    #disp_gts = [disp_gt, disp_gt, disp_gt, disp_gt, disp_gt, disp_gt]
    #loss = model_loss_test(disp_ests, disp_gt, mask)
    disp_est = inference(imgL, imgR, model, n_iter=20)
    disp_ests = [disp_est]
    loss = F.l1_loss(disp_est, disp_gt)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_est, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    train()



