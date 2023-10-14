# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset, ODIRDataset, save_batch_images, TestDataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.asymmetric_loss import AsymmetricLossOptimized
import models_vit
from ODIR_model import ODIRmodel
from generate_predictions_csv import save_predictions
from engine_finetune import train_one_epoch, evaluate
from torch.utils.data import Dataset, DataLoader

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--task', default='',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/jupyter/Mor_DR_data/data/data/IDRID/Disease_Grading/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=8, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    low_quality_files = [
    "2174_right.jpg",
    "2175_left.jpg",
    "2176_left.jpg",
    "2177_left.jpg",
    "2177_right.jpg",
    "2178_right.jpg",
    "2179_left.jpg",
    "2179_right.jpg",
    "2180_left.jpg",
    "2180_right.jpg",
    "2181_left.jpg",
    "2181_right.jpg",
    "2182_left.jpg",
    "2182_right.jpg",
    "2957_left.jpg",
    "2957_right.jpg",
    "2340_lef.jpg",
    "1706_left.jpg",
    "1710_right.jpg",
    "4522_left.jpg",
    "1222_right.jpg", 
    "1260_left.jpg", 
    "2133_right.jpg", 
    "240_left.jpg",
    "240_right.jpg",
    "150_left.jpg", 
    "150_right.jpg",
    ]
    # Manual found low quality: 2340 left, 1706_left, 1710_right, 4522_left, 1222_right, 1260_left
    # 2133_right, 240_left, 240_right, 150_left, 150_right


    
    disease_columns = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    original_df = pd.read_excel('/home/scur0556/ODIR2019/data/ODIR-5K_Training_Annotations(Updated)_V2.xlsx')
    original_df = original_df.drop(columns=['Left-Diagnostic Keywords', 'Right-Diagnostic Keywords'])
    print(original_df[disease_columns].sum() / len(original_df) * 100)
    balanced_df = pd.read_csv('/home/scur0556/ODIR2019/data/balanced_df.csv')
    balanced_df = balanced_df.drop(columns=['Left-Diagnostic Keywords', 'Right-Diagnostic Keywords'])
    
    

    
    valid_rows = []
    for idx, row in original_df.iterrows():
        left_img_name = os.path.join('data/cropped_ODIR-5K_Training_Dataset', row['Left-Fundus'])
        right_img_name = os.path.join('data/cropped_ODIR-5K_Training_Dataset', row['Right-Fundus'])

        if left_img_name not in low_quality_files and right_img_name not in low_quality_files:
            valid_rows.append(row)
    original_df = pd.DataFrame(valid_rows)
    

    #skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    #for fold, (train_idx, val_idx) in enumerate(skf.split(original_df)):
    #print("train set length", len(train_df))
    #print("val set length", len(validation_df))
    train_df, validation_df = train_test_split(original_df, test_size=0.2, random_state=42)
   
    dataset_train = ODIRDataset(train_df, '/home/scur0556/ODIR2019/data/cropped_ODIR-5K_Training_Dataset', is_train=True, args=args)
    dataset_val = ODIRDataset(validation_df, '/home/scur0556/ODIR2019/data/cropped_ODIR-5K_Training_Dataset', is_train=False, args=args)
    

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir+args.task)
    else:
        log_writer = None


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        sampler=sampler_val, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    save_batch_images(data_loader_train, num_images=8, filename="batch_visualization.png")
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    base_vit_model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

   
    # Instantiate a pre-trained Vision Transformer model.
    base_vit_model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,  # May not be necessary depending on your needs.
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # Extract the model's weights for fine-tuning, if applicable.
    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        
        # Potentially modify checkpoint as per your requirements (e.g., remove classification head)
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint['model']:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint['model'][k]

        # Load pre-trained weights into the Vision Transformer model.
        # Note: Adaptation might be needed depending on checkpoint format.
        base_vit_model.load_state_dict(checkpoint['model'], strict=False)

    
    model = ODIRmodel(base_vit_model=base_vit_model, num_classes=args.nb_classes) 
    print('odir', model)
    

    model.to(device)

    # for param in model.base_vit_model.parameters():
    #     param.requires_grad = False

    # for i, block in enumerate(model.base_vit_model.blocks):
    #     for param in block.parameters():
    #         param.requires_grad = i >= (len(model.base_vit_model.blocks) - 2)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    # if mixup_fn is not None:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif args.smoothing > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    # else:

    samples = original_df.shape[0]
    df_diseases = original_df[disease_columns]
    class_weights = list(samples / (8 * np.sum(df_diseases, axis=0)))
    print(class_weights)
    class_weights = torch.tensor(class_weights).to(device)

    #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    # criterion = torch.nn.BCEWithLogitsLoss()
    # https://github.com/Alibaba-MIIL/ASL/issues/22#issuecomment-736721770
    # use link above to adjust aysmmetric loss to focal loss 
    criterion = AsymmetricLossOptimized()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats,auc_roc = evaluate(data_loader_val, model, device, args.task, epoch=0, mode='test',num_class=args.nb_classes)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_auc = 0.0
    best_final_score = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

       
        val_loss, final_score, kappa, f1, auc = evaluate(model, data_loader_val, device, criterion)
    
        # Log validation metrics
        print(f"Validation Loss: {val_loss:.4f}, Final Score: {final_score:.4f}, Kappa: {kappa:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        # Check if this is the best model so far
        if final_score > best_final_score:
            best_final_score = final_score
            
            if args.output_dir:
                # not sure how this works with distributed training so TODO
                # misc.save_model(
                #     args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                #     loss_scaler=loss_scaler, epoch=epoch)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_final_score': best_final_score,
                }
                checkpoint_path = os.path.join(args.output_dir, 'best_model_checkpoint.pth')
                torch.save(checkpoint, checkpoint_path)
                print("Model checkpoint saved at", checkpoint_path)
        

            
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    from generate_predictions_csv import save_predictions
    checkpoint_path = os.path.join(args.output_dir, 'best_model_checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test_loader = DataLoader(TestDataset('/home/scur0556/ODIR2019/data/cropped_ODIR-5K_Testing_Images', is_train=False, args=args), batch_size=16, shuffle=False)
    output_path = os.path.join(args.output_dir, 'prob_predictions.csv')
    save_predictions(model, test_loader, device, output_path, logit_output=True)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
