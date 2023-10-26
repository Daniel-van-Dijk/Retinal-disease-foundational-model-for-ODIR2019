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
from torchvision import transforms, models
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os




from torchvision import datasets
from tqdm.notebook import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

import csv
import torch.nn.functional as F

import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from sklearn.model_selection import train_test_split, GroupShuffleSplit

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset, ODIRDataset, TestDataset, Merge_valset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from evaluate import ODIR_Metrics
import models_vit
from ODIR_model import ODIRmodel

from engine_finetune import train_one_epoch, evaluate


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
    
    #test_loader = DataLoader(TestDataset('/home/scur0556/ODIR2019/data/cropped_ODIR-5K_Testing_Images', is_train=False, args=args), batch_size=16, shuffle=False)
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
 

    
    disease_columns = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    original_df = pd.read_excel('/home/scur0556/ODIR2019/data/ODIR-5K_Training_Annotations(Updated)_V2.xlsx')
    original_df = original_df.drop(columns=['Left-Diagnostic Keywords', 'Right-Diagnostic Keywords'])
    
    expanded_df = pd.read_csv("/home/scur0556/ODIR2019/data/expanded_sheet_annotations_fixed.csv")
    expanded_df = expanded_df.drop(columns=['Left-Diagnostic Keywords', 'Right-Diagnostic Keywords'])

    # print(len(original_df))
    # valid_rows = []
    # for idx, row in original_df.iterrows():
    #     if row['Left-Fundus'] not in low_quality_files and row['Right-Fundus'] not in low_quality_files:
    #         valid_rows.append(row)
    # original_df = pd.DataFrame(valid_rows)
    low_quality_files_set = set(low_quality_files)

   
    original_df = original_df[~original_df['Left-Fundus'].isin(low_quality_files_set) & ~original_df['Right-Fundus'].isin(low_quality_files_set)]

    valid_ids = original_df['ID'].unique().tolist()
    expanded_df = expanded_df[expanded_df['ID'].isin(valid_ids)]
    
    unique_ids = original_df['ID'].unique().tolist()
    shuffled_ids = pd.Series(unique_ids).sample(frac=1, random_state=42).tolist()

    train_size = int(0.8 * len(shuffled_ids))
   
    val_patient_ids = shuffled_ids[train_size:]

    validation_df = original_df[original_df['ID'].isin(val_patient_ids)]

    expanded_validation_df = expanded_df[expanded_df['ID'].isin(val_patient_ids)]

    expanded_validation_df = expanded_validation_df.copy()
    expanded_validation_df['Eye_Type'] = expanded_validation_df.apply(
    lambda row: 'Left' if pd.notna(row['Left-Fundus']) else 'Right', axis=1)

    expanded_validation_df = expanded_validation_df.sort_values(by=['ID', 'Eye_Type']).reset_index(drop=True)
    expanded_validation_df = expanded_validation_df.drop(columns=['Eye_Type'])
    validation_df = validation_df.sort_values(by=['ID']).reset_index(drop=True)

    #print(expanded_validation_df)

    val_loader = Merge_valset(expanded_validation_df, '/home/scur0556/ODIR2019/data/cropped_ODIR-5K_Training_Dataset', is_train=False, args=args)
    



    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(device)


    base_vit_model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    model = ODIRmodel(base_vit_model=base_vit_model, num_classes=args.nb_classes)
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    checkpoint_path = '/home/scur0556/ODIR2019/src/models/foundational_model/run_4234990/best_score_checkpoint.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device) 
    state_dict = checkpoint['model_state_dict']
    model_sd_keys = set(model.state_dict().keys())
    ckpt_sd_keys = set(state_dict.keys())

    missing_keys = model_sd_keys - ckpt_sd_keys
    unexpected_keys = ckpt_sd_keys - model_sd_keys

    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    model.load_state_dict(state_dict)

    model.eval()
    data_loader_val = DataLoader(val_loader, batch_size=16, shuffle=False)
    # check logit_output param
    _, predictions = create_predictions(model, data_loader_val, device, logit_output=True)
    combined_patient_predictions = combine_predictions(predictions)

    all_preds = (combined_patient_predictions > 0.5).astype(np.float32)
    all_labels = validation_df.iloc[:, 5:].values.astype(np.float32)

    kappa, f1, auc, final_score = ODIR_Metrics(all_labels, all_preds)
    print('kappa: ', kappa)
    print('f1: ', f1)
    print('auc: ', auc)
    print('final_score: ', final_score)

    test_loader = DataLoader(TestDataset('/home/scur0556/ODIR2019/data/cropped_ODIR-5K_Testing_Images', is_train=False, args=args), batch_size=16, shuffle=False)
    save_combined_predictions(model, test_loader, device, 'saved_combined2.csv', logit_output=True)

def create_predictions(model, dataloader, device, logit_output=True):
    model.eval()
    all_ids = []
    all_probs = []

    with torch.no_grad():
        for (images_left, images_right, image_ids) in dataloader: 
            images_left, images_right = images_left.to(device), images_right.to(device)

            output_left = model(images_left)
            output_right = model(images_right)
            if logit_output:
                output_left = torch.sigmoid(output_left)
                output_right = torch.sigmoid(output_right)

            all_ids.extend(image_ids.cpu().numpy())
            
            for l, r in zip(output_left.cpu().numpy(), output_right.cpu().numpy()):
                all_probs.append(l)
                all_probs.append(r)

    all_probs = np.vstack(all_probs)

    return all_ids, all_probs


def combine_predictions(probs, threshold=0.5):
    combined_probs = []
    for i in range(0, len(probs), 2):
        left_eye_prob = probs[i]
        right_eye_prob = probs[i+1]

        max_prob = np.maximum(left_eye_prob, right_eye_prob)

        # If any disease is above the threshold, set the "Normal" flag to 0
        if np.any(max_prob[1:] > threshold):
            max_prob[0] = 0

        combined_probs.append(max_prob)

    return np.array(combined_probs)


def save_combined_predictions(model, dataloader, device, output_file, logit_output=True, threshold=0.5):
    all_ids, all_probs = create_predictions(model, dataloader, device, logit_output)
    combined_probs = combine_predictions(all_probs, threshold)

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])

        for idx, prob in zip(all_ids, combined_probs):
            writer.writerow([idx] + list(prob))



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)