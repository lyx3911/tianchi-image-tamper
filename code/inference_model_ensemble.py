import os
import io
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from models.mvssnet import get_mvss
from models.unet_model import Ringed_Res_Unet
from common.tools import inference_single
import segmentation_models_pytorch as smp

import argparse

from dataset import ManiDataset
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--root', type=str, default='/data1/datasets/Image-Manipulation-Detection/test/')
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--th', type=float, default=0.5)
    parser.add_argument('--weights', type=str, default='./ckpt/mvssnet_casia.pt')
    parser.add_argument('--save-dir', type=str, default='./images/')
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
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
        print("load model :{}".format(pretrained_ckpt))
    elif model_type == 'rrunet': 
        model = Ringed_Res_Unet(n_channels=3, n_classes=1)
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
        print("load model :{}".format(pretrained_ckpt))
    elif model_type == 'unet':
        model = smp.Unet('efficientnet-b5', classes=1, activation='sigmoid')
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
        print("load model :{}".format(pretrained_ckpt))
    elif model_type == 'linknet':
        model = smp.Linknet('efficientnet-b5', classes=1, activation='sigmoid')
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
        print("load model :{}".format(pretrained_ckpt))
        
    model.load_state_dict(checkpoint, strict=True)
    model = model.cuda()
    model.eval()
    return model

def inference_single_rru(model, img, th=0):
    transform_pil = transforms.Compose([
        transforms.ToPILImage(),
    ])
    img = img.cuda().view(-1, img.shape[0], img.shape[1], img.shape[2])
    # print(img)
    with torch.no_grad():
        seg = model(img)
        # print(seg)
        seg = torch.sigmoid(seg).detach().cpu()
        # print(seg)
        
        if torch.isnan(seg).any() or torch.isinf(seg).any():
                max_score = 0.0
        else:
            max_score = torch.max(seg).numpy()
        seg = [np.array(transform_pil(seg[i])) for i in range(len(seg))]
        # print(seg)
    
    if len(seg) != 1:
        pdb.set_trace()
    else:
        fake_seg = seg[0]
    if th == 0:
        return fake_seg, max_score
    # fake_seg = 255.0 * (fake_seg > 255 * th)
    # fake_seg = fake_seg.astype(np.uint8)

    # print(fake_seg.shape)
    return fake_seg, max_score

def inference_single_unet(model, img, th=0):
    transform_pil = transforms.Compose([
        transforms.ToPILImage(),
    ])
    torch_resize = transforms.Resize([512, 512])
    img = torch_resize(img)
    
    img = img.cuda().view(-1, img.shape[0], img.shape[1], img.shape[2])
    # print(img)
    with torch.no_grad():
        seg = model(img)
        # print(seg)
        seg = seg.detach().cpu()
        # print(seg)
        
        if torch.isnan(seg).any() or torch.isinf(seg).any():
                max_score = 0.0
        else:
            max_score = torch.max(seg).numpy()
        seg = [np.array(transform_pil(seg[i])) for i in range(len(seg))]
        # print(seg)
    
    if len(seg) != 1:
        pdb.set_trace()
    else:
        fake_seg = seg[0]
    if th == 0:
        return fake_seg, max_score
    # fake_seg = 255.0 * (fake_seg > 255 * th)
    # fake_seg = fake_seg.astype(np.uint8)

    # print(fake_seg.shape)
    return fake_seg, max_score

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

def slide_window_val_unet(model, valloader, th=0.5):
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
        loss = dice_loss(seg, label).item()
   
        losses.update(loss, 1)  
        f1, iou = calculate_batch_score(seg, label, th)
        scores.update(f1, 1)
        ious.update(iou, 1) 
        
    return losses.avg, scores.avg, ious.avg

def TTA_val(model, valloader, th=0.5, alpha=0.3):
    losses = AverageMeter()
    scores = AverageMeter()
    ious = AverageMeter()
    model.eval()
    
    dice_loss = DiceLoss()   
    # dice_loss = PixelClsLoss() 
    
    for img, label in tqdm(valloader):
        img, label = img.cuda(), label.cuda() #[3,h,w]
        # print(img.shape, label.shape)
        
        # 滑窗计算
        with torch.no_grad():
            img_h, img_w = img.shape[-2], img.shape[-1]
            col, row = cal_sliding_params(img_h, img_w)
            imgs = img_slide_window(img, col, row)
            # print(imgs.shape)
            seg = model(imgs)
        # seg = torch.sigmoid(seg)
        seg = merge_slides_result(seg, col, row, img.shape)
        loss = dice_loss(seg, label).item()
        
        # 直接resize 计算
        transform = transforms.Compose([
            transforms.Resize([512,512])
        ])
        invtransform = transforms.Compose([
            transforms.Resize([img.shape[-2], img.shape[-1]])
        ]) 
        with torch.no_grad():
            seg_resize = model(transform(img))
            seg_resize = invtransform(seg_resize)
        seg = seg*alpha + seg_resize*(1-alpha)       
        
        losses.update(loss, 1)  
        f1, iou = calculate_batch_score(seg, label, th)
        scores.update(f1, 1)
        ious.update(iou, 1) 
        
    return losses.avg, scores.avg, ious.avg


def TTA_inference_single_unet(model, img, th=0, alpha=0.8):
    transform_pil = transforms.Compose([
        transforms.ToPILImage(),
    ])
    img = img.cuda().view(-1, img.shape[0], img.shape[1], img.shape[2])
    # print(img)
    
    # 滑窗检测
    with torch.no_grad():
        img_h, img_w = img.shape[-2], img.shape[-1]
        col, row = cal_sliding_params(img_h, img_w)
        imgs = img_slide_window(img, col, row)
        # print(imgs.shape)
        seg = model(imgs)
        # seg = torch.sigmoid(seg)
        seg = merge_slides_result(seg, col, row, img.shape)
    
    transform = transforms.Compose([
        transforms.Resize([512,512])
    ])
    invtransform = transforms.Compose([
        transforms.Resize([img.shape[-2], img.shape[-1]])
    ]) 
    with torch.no_grad():
        seg_resize = model(transform(img))
        seg_resize = invtransform(seg_resize)
    seg = seg*alpha + seg_resize*(1-alpha)
    seg = seg.detach().cpu()    
    
    seg = seg - seg.min()
    seg = seg / seg.max()
        
    if torch.isnan(seg).any() or torch.isinf(seg).any():
        max_score = 0.0
    else:
        max_score = torch.max(seg).numpy()
    seg = [np.array(transform_pil(seg[i])) for i in range(len(seg))]

    
    if len(seg) != 1:
        pdb.set_trace()
    else:
        fake_seg = seg[0]
    if th == 0:
        return fake_seg, max_score
    
    # fake_seg = 255.0 * (fake_seg > 255 * th)
    # fake_seg = fake_seg.astype(np.uint8)

    # print(fake_seg.shape)
    return fake_seg, max_score


if __name__ == '__main__':
    args = parse_args()
    weight1 = "./work_dir/2859-unet-5fold0-aug/weights/best.pth.tar" # fold 4
    weight2 = "./work_dir/2859-unet-5fold1-aug/weights/best.pth.tar" # fold 1
    weight3 = "./work_dir/2859-unet-5fold2-aug/weights/best.pth.tar" # fold2
    weight4 = "./work_dir/2859-unet-5fold3-aug/weights/best.pth.tar"
    model1 = init_model('unet', weight1)
    model2 = init_model('unet', weight2)
    model3 = init_model('unet', weight3)
    model4 = init_model('unet', weight4)
    
    # init dataset
    testset = ManiDataset([args.root], ['test.txt'], mode='test', resize=True)
    
    for img, img_name, fake_size in tqdm(testset):
        # print(img.shape, img_name, fake_size)
        seg1, max_score1 = TTA_inference_single_unet(model1, img, th=args.th, alpha=0.3)
        seg2, max_score2 = TTA_inference_single_unet(model2, img, th=args.th, alpha=0.3)
        seg3, max_score3 = TTA_inference_single_unet(model3, img, th=args.th, alpha=0.3)
        seg4, max_score4 = TTA_inference_single_unet(model4, img, th=args.th, alpha=0.3)
        
        seg1 = cv2.resize(seg1, (fake_size[1], fake_size[0]))
        seg2 = cv2.resize(seg2, (fake_size[1], fake_size[0]))
        seg3 = cv2.resize(seg3, (fake_size[1], fake_size[0]))
        seg4 = cv2.resize(seg4, (fake_size[1], fake_size[0]))
        # print(seg1.shape)
        seg = seg1*0.25 + seg2*0.25 + seg3*0.25 + seg4*0.25
        
        seg = 255.0 * (seg > 255 * args.th)
        seg = seg.astype(np.uint8)
        _, seg = cv2.threshold(seg, int(255*args.th), 255, cv2.THRESH_BINARY)        
        cv2.imwrite("{}/{}.png".format(args.save_dir, img_name), seg)