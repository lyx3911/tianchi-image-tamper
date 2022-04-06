import os
import io
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from models.mvssnet import get_mvss
from models.unet_model import Ringed_Res_Unet, Ringed_Res_Unet_Slim
import segmentation_models_pytorch as smp
# from common.tools import inference_single
from common.utils import calculate_pixel_f1, calculate_img_score, AverageMeter
from loss import ClsLoss, DiceLoss, PixelClsLoss, EdgeLoss
from loss import *

import torch.utils.data as data
import argparse
from tqdm import tqdm
from dataset import ManiDataset, ManiDatasetAug

from loguru import logger

def parse_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--root', type=str, default='/data1/datasets/Image-Manipulation-Detection/train/')
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--pretrained-ckpt', type=str, default=None)
    parser.add_argument('--th', type=float, default=0.5)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=1)
    # parser.add_argument("--model_name", type=str, help="Path to the pretrained model", default="ckpt/mvssnet.pth")
    parser.add_argument('--work-dir', type=str, default='./work_dir/')
    args = parser.parse_args()
    return args

def init_model(model_type, pretrained_ckpt=None):
    if model_type == 'mvssnet':
        model = get_mvss(backbone='resnet50',
                     pretrained_base=True,
                     nclass=1,
                     sobel=True,
                     constrain=True,
                     n_input=3,
                     )
        # TODO: initialize with pretrained_ckpt
        # checkpoint = torch.load("./ckpt/mvssnet_tianchi.pt", map_location='cpu')
    elif model_type == 'rrunet': 
        model = Ringed_Res_Unet(n_channels=3, n_classes=1)
        # checkpoint = torch.load("./work_dir/rru-diceloss-fold4/weights/last.pth.tar", map_location='cpu')
        # model.load_state_dict(checkpoint, strict=True)
    elif model_type == 'rrunet-slim':
        model = Ringed_Res_Unet_Slim(n_channels=3, n_classes=1)
    elif model_type == 'unet':
        model = smp.Unet('tu-eca_swinnext26ts_256', classes=1, activation='sigmoid')
    elif model_type == 'linknet':
        model = smp.Linknet('efficientnet-b5', classes=1, activation='sigmoid')
    
    if pretrained_ckpt != None:
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)        
    model = model.cuda()
    return model
    
def train_mvss(model, trainloader, optimizer):
    losses = AverageMeter()
    f1_score = AverageMeter()
    ious = AverageMeter()
    
    model.train()
    
    dice_loss = DiceLoss()
    cls_loss = ClsLoss()
    alpha = 0.16
    beta = 0.04
    edge_loss = EdgeLoss()
    pixel_bceloss = PixelClsLoss()
    
    for img, label in tqdm(trainloader):
        img, label = img.cuda(), label.cuda()
        # print(img.shape, label.shape)
        edge, seg = model(img)
        # print(edge.shape, seg.shape)
        seg = torch.sigmoid(seg)
        loss = alpha*dice_loss(seg, label) + beta*cls_loss(seg, label) + (1-alpha-beta)*edge_loss(edge, label)
        # loss = pixel_bceloss(seg, label)
        f1, iou = calculate_batch_score(seg, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
        losses.update(loss.item(), img.shape[0])
        f1_score.update(f1, img.shape[0])
        ious.update(iou, img.shape[0])
        
    return losses.avg, f1_score.avg, ious.avg
    
def calculate_iou(segs, labels, eps=1e-8):
    intersaction = segs * labels
    iou = intersaction.sum()/(segs.sum()+labels.sum()-intersaction.sum() + eps)
    return iou
    
def calculate_batch_score(segs, labels):
    batch_size = segs.shape[0]
    batch_f1, batch_iou = 0.0, 0.0
    for i in range(batch_size):
        pd = segs[i]
        gt = labels[i]
        fake_seg = pd.detach().cpu().numpy()
        fake_gt = gt.detach().cpu().numpy()
        fake_seg = np.where(fake_seg<0.5, 0.0, 1.0)
        # print(fake_seg.shape, fake_gt.shape)
        f1, p, r = calculate_pixel_f1(fake_seg.flatten(),fake_gt.flatten())
        batch_f1 += f1
        iou = calculate_iou(fake_seg.flatten(),fake_gt.flatten())
        batch_iou += iou
    return batch_f1 / batch_size, batch_iou / batch_size
        
        
def val_mvss(model, valloader):
    losses = AverageMeter()
    scores = AverageMeter()
    ious = AverageMeter()
    model.eval()
    
    dice_loss = DiceLoss()
    alpha = 0.16
    cls_loss = ClsLoss()
    beta = 0.04
    edge_loss = EdgeLoss()
    
    pixel_bceloss = PixelClsLoss()
    
    for img, label in tqdm(valloader):
        img, label = img.cuda(), label.cuda()
        # print(img.shape, label.shape)
        with torch.no_grad():
            edge, seg = model(img)
        seg = torch.sigmoid(seg)
        loss = alpha*dice_loss(seg, label).item() + beta*cls_loss(seg,label).item() + (1-alpha-beta)*edge_loss(edge, label).item()
        # loss = pixel_bceloss(seg, label).item()     
        losses.update(loss, 1)  
        f1, iou = calculate_batch_score(seg, label)
        scores.update(f1, 1)
        ious.update(iou, 1) 
        
    return losses.avg, scores.avg, ious.avg    

def train_rru(model, trainloader, optimizer):
    losses = AverageMeter()
    f1_score = AverageMeter()
    ious = AverageMeter()
    
    model.train()
    
    dice_loss = DiceLoss()
    bce_loss = PixelClsLoss()
    
    for img, label in tqdm(trainloader):
        img, label = img.cuda(), label.cuda()
        # print(img.shape, label.shape)
        seg = model(img)
        # print(edge.shape, seg.shape)
        seg = torch.sigmoid(seg)
        loss = dice_loss(seg, label)*0.3 + bce_loss(seg, label)*0.7

        f1, iou = calculate_batch_score(seg, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
        losses.update(loss.item(), img.shape[0])
        f1_score.update(f1, img.shape[0])
        ious.update(iou, img.shape[0])
        
    return losses.avg, f1_score.avg, ious.avg

def val_rru(model, valloader):
    losses = AverageMeter()
    scores = AverageMeter()
    ious = AverageMeter()
    model.eval()
    
    dice_loss = DiceLoss()   
    # dice_loss = PixelClsLoss() 
    
    for img, label in tqdm(valloader):
        img, label = img.cuda(), label.cuda()
        # print(img.shape, label.shape)
        with torch.no_grad():
            seg = model(img)
        seg = torch.sigmoid(seg)
        loss = dice_loss(seg, label).item()
   
        losses.update(loss, 1)  
        f1, iou = calculate_batch_score(seg, label)
        scores.update(f1, 1)
        ious.update(iou, 1) 
        
    return losses.avg, scores.avg, ious.avg   

def train_unet(model, trainloader, optimizer):
    losses = AverageMeter()
    f1_score = AverageMeter()
    ious = AverageMeter()
    
    global best_score
    
    model.train()
    
    dice_loss = DiceLoss()
    bce_loss = PixelClsLoss()
    
    i = 0
    
    for img, label in tqdm(trainloader):
        img, label = img.cuda(), label.cuda()
        # print(img.shape, label.shape)
        seg = model(img)
        # # print(edge.shape, seg.shape)
        # seg = torch.sigmoid(seg)
        loss = dice_loss(seg, label)*0.3 + bce_loss(seg, label)*0.7
        # loss = lovasz_softmax(seg, label)

        f1, iou = calculate_batch_score(seg, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
        losses.update(loss.item(), img.shape[0])
        f1_score.update(f1, img.shape[0])
        ious.update(iou, img.shape[0])
        
    return losses.avg, f1_score.avg, ious.avg

def val_unet(model, valloader):
    losses = AverageMeter()
    scores = AverageMeter()
    ious = AverageMeter()
    model.eval()
    
    dice_loss = DiceLoss()   
    # dice_loss = PixelClsLoss() 
    
    for img, label in tqdm(valloader):
        img, label = img.cuda(), label.cuda()
        # print(img.shape, label.shape)
        with torch.no_grad():
            seg = model(img)
        # seg = torch.sigmoid(seg)
        loss = dice_loss(seg, label).item()
   
        losses.update(loss, 1)  
        f1, iou = calculate_batch_score(seg, label)
        scores.update(f1, 1)
        ious.update(iou, 1) 
        
    return losses.avg, scores.avg, ious.avg   

def cal_sliding_params(img_h, img_w):
    # 计算需要裁剪成几块
    col, row = 1, 1
    while (512*col - (col-1)*128) < img_h:
        col += 1
    while (512*row - (row-1)*128) < img_w:
        row += 1
    return col, row

def img_slide_window(img, col, row):
    imgs = []
    # 计算 overlape
    delta_x, delta_y = 0, 0
    if row > 1:
        delta_x = int((row*512-img.shape[-1])/(row-1))
    if col > 1:
        delta_y = int((col*512-img.shape[-2])/(col-1))
        
    for i in range(col):
        for j in range(row):
            begin_h = 512*i - max(0, i)*delta_y
            begin_w = 512*j - max(0, j)*delta_x
            
            if begin_h + 512 > img.shape[-2]:
                begin_h = img.shape[-2] - 512
            if begin_w + 512 > img.shape[-1]:
                begin_w = img.shape[-1] - 512
            slide = img[:, :, begin_h:begin_h+512, begin_w:begin_w+512].squeeze(0)
            imgs.append(slide)
            # print(begin_h, begin_w, begin_h+512, begin_w+512, img.shape)
    return torch.stack(imgs, dim=0)

def merge_slides_result(segs, col, row, img_shape):
    count = torch.zeros([1, img_shape[2], img_shape[3]]).cuda()
    seg = torch.zeros([1, img_shape[2], img_shape[3]]).cuda()
    
    # 计算 overlape
    delta_x, delta_y = 0, 0
    if row > 1:
        delta_x = int((row*512-img_shape[-1])/(row-1))
    if col > 1:
        delta_y = int((col*512-img_shape[-2])/(col-1))
        
    # print(col, row)
    for i in range(col):
        for j in range(row):
            begin_h = 512*i - max(0, i)*delta_y
            begin_w = 512*j - max(0, j)*delta_x
            
            if begin_h + 512 > img_shape[-2]:
                begin_h = img_shape[-2] - 512
            if begin_w + 512 > img_shape[-1]:
                begin_w = img_shape[-1] - 512
            seg[:, begin_h:begin_h+512, begin_w:begin_w+512] += segs[i*row+j]
            count[:, begin_h:begin_h+512, begin_w:begin_w+512] += 1.0
    seg = seg / count
    return seg.unsqueeze(0)

def slide_window_val_unet(model, valloader):
    losses = AverageMeter()
    scores = AverageMeter()
    ious = AverageMeter()
    model.eval()
    
    dice_loss = DiceLoss()   
    # dice_loss = PixelClsLoss() 
    
    for img, label in tqdm(valloader):
        img, label = img.cuda(), label.cuda() #[3,h,w]
        # print(img.shape, label.shape)
        with torch.no_grad():
            img_h, img_w = img.shape[-2], img.shape[-1]
            col, row = cal_sliding_params(img_h, img_w)
            imgs = img_slide_window(img, col, row)
            # print(imgs.shape)
            seg = model(imgs)
        # seg = torch.sigmoid(seg)
        seg = merge_slides_result(seg, col, row, img.shape)
        # print(seg.shape, label.shape)
        loss = dice_loss(seg, label).item()
        # print(loss)
        # exit()
   
        losses.update(loss, 1)  
        f1, iou = calculate_batch_score(seg, label)
        scores.update(f1, 1)
        ious.update(iou, 1) 
    # print(losses.avg)    
    return losses.avg, scores.avg, ious.avg

def save_checkpoint(state, work_dir, name):
    filepath = os.path.join(work_dir, "weights/", name + '.pth.tar')
    torch.save(state, filepath)
 
def set_work_dir(work_dir):
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
        os.mkdir(os.path.join(work_dir, "weights/"))
    logger.add(os.path.join(work_dir, "info.log"))
    logger.info("Training begin")
            
if __name__ == '__main__':
    args = parse_args()
    
    set_work_dir(args.work_dir)
    
    # init model
    model = init_model(args.model, args.pretrained_ckpt)
    
    root = ["/data1/datasets/Image-Manipulation-Detection/train/", ]
    split = ["train-split0.txt"]
    
    #init dataset
    trainset = ManiDatasetAug(root, split=split, w=512, h=512, mode='train')
    trainloader = data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=4)
    valset = ManiDataset([args.root], split=["val-split0.txt"], w=512, h=512, mode='val')
    valloader = data.DataLoader(valset, batch_size=args.batchsize*2, shuffle=False, num_workers=4)
    # print(len(trainset),len(valset))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9,0.999), eps=1e-08, weight_decay=0.0)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9,0.999), eps=1e-08, weight_decay=0.005)
    
    global best_score
    best_score = 0.0
    for epoch in range(args.epoch):
        # epoch_loss, train_f1, train_iou = train_rru(model, trainloader, optimizer)
        epoch_loss, train_f1, train_iou = train_unet(model, trainloader, optimizer) 

        eval_loss, eval_f1, eval_iou = val_unet(model, valloader)
        # eval_loss, eval_f1, eval_iou = val_rru(model, valloader)
        # eval_loss, eval_f1, eval_iou = slide_window_val_unet(model, valloader)

        logger.info("epoch: {}, train loss: {}, train f1: {}, train iou: {}, train score: {}, val loss: {}, val f1: {}, val iou: {}, val score: {}".format(epoch, epoch_loss, train_f1, train_iou, train_iou+train_f1, eval_loss, eval_f1, eval_iou, eval_iou+eval_f1))
        save_checkpoint(model.state_dict(), args.work_dir, "last") 
        
        if(eval_f1 + eval_iou > best_score):
            best_score = eval_f1 + eval_iou 
            save_checkpoint(model.state_dict(), args.work_dir, "best")   
            logger.info("save_checkpoint, best_score: {}".format(best_score))