# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import math
import sys
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score,multilabel_confusion_matrix
from pycm import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics




def misc_measures(confusion_matrix):
    
    acc = []
    sensitivity = []
    specificity = []
    precision = []
    G = []
    F1_score_2 = []
    mcc_ = []
    
    for i in range(1, confusion_matrix.shape[0]):
        cm1=confusion_matrix[i]
        acc.append(1.*(cm1[0,0]+cm1[1,1])/np.sum(cm1))
        sensitivity_ = 1.*cm1[1,1]/(cm1[1,0]+cm1[1,1])
        sensitivity.append(sensitivity_)
        specificity_ = 1.*cm1[0,0]/(cm1[0,1]+cm1[0,0])
        specificity.append(specificity_)
        precision_ = 1.*cm1[1,1]/(cm1[1,1]+cm1[0,1])
        precision.append(precision_)
        G.append(np.sqrt(sensitivity_*specificity_))
        F1_score_2.append(2*precision_*sensitivity_/(precision_+sensitivity_))
        mcc = (cm1[0,0]*cm1[1,1]-cm1[0,1]*cm1[1,0])/np.sqrt((cm1[0,0]+cm1[0,1])*(cm1[0,0]+cm1[1,0])*(cm1[1,1]+cm1[1,0])*(cm1[1,1]+cm1[0,1]))
        mcc_.append(mcc)
        
    acc = np.array(acc).mean()
    sensitivity = np.array(sensitivity).mean()
    specificity = np.array(specificity).mean()
    precision = np.array(precision).mean()
    G = np.array(G).mean()
    F1_score_2 = np.array(F1_score_2).mean()
    mcc_ = np.array(mcc_).mean()
    
    return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_





def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, ((img_left, img_right), targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        img_left = img_left.to(device, non_blocking=True)
        img_right = img_right.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        with torch.cuda.amp.autocast():
            outputs = model(img_left, img_right)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate_old(data_loader, model, device, task, epoch, mode, num_class):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if not os.path.exists(task):
        os.makedirs(task)

    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []
    
    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        true_label=F.one_hot(target.to(torch.int64), num_classes=num_class)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
            prediction_softmax = nn.Softmax(dim=1)(output)
            _,prediction_decode = torch.max(prediction_softmax, 1)
            _,true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
            true_label_onehot_list.extend(true_label.cpu().detach().numpy())
            prediction_list.extend(prediction_softmax.cpu().detach().numpy())

        acc1,_ = accuracy(output, target, topk=(1,2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)
    confusion_matrix = multilabel_confusion_matrix(true_label_decode_list, prediction_decode_list,labels=[i for i in range(num_class)])
    acc, sensitivity, specificity, precision, G, F1, mcc = misc_measures(confusion_matrix)
    
    auc_roc = roc_auc_score(true_label_onehot_list, prediction_list,multi_class='ovr',average='macro')
    auc_pr = average_precision_score(true_label_onehot_list, prediction_list,average='macro')          
            
    metric_logger.synchronize_between_processes()
    
    print('Sklearn Metrics - Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f} MCC: {:.4f}'.format(acc, auc_roc, auc_pr, F1, mcc)) 
    results_path = task+'_metrics_{}.csv'.format(mode)
    with open(results_path,mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data2=[[acc,sensitivity,specificity,precision,auc_roc,auc_pr,F1,mcc,metric_logger.loss]]
        for i in data2:
            wf.writerow(i)
            
    
    if mode=='test':
        cm = ConfusionMatrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
        cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=True,plot_lib="matplotlib")
        plt.savefig(task+'confusion_matrix_test.jpg',dpi=600,bbox_inches ='tight')
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},auc_roc




#calculate kappa, F-1 socre and AUC value
def ODIR_Metrics(gt_data, pr_data):
    """ function from ODIR2019 challenge """
    th = 0.5
    gt = gt_data.flatten()
    pr = pr_data.flatten()
    kappa = metrics.cohen_kappa_score(gt, pr>th)
    f1 = metrics.f1_score(gt, pr>th, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa+f1+auc)/3.0
    return kappa, f1, auc, final_score

@torch.no_grad()
def evaluate(model, dataloader, device, criterion):
    model.eval()
    all_labels = []
    all_logits = []
    val_loss = 0
    with torch.no_grad():
        for (images_left, images_right), labels in dataloader:
            images_left, images_right = images_left.to(device), images_right.to(device)
            labels = labels.to(device)

            logits = model(images_left, images_right)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_logits = np.vstack(all_logits)

    all_preds = (all_logits > 0.5).astype(np.float32)
    # val_loss, average_score, kappa, f1, auc, val_confusion, val_accuracy
    kappa, f1, auc, final_score = ODIR_Metrics(all_labels, all_preds)

    return val_loss / len(dataloader), final_score, kappa, f1, auc