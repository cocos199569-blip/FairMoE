# %%
# import dgl
# import ipdb
import time
import argparse
import numpy as np
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import warnings
import os

warnings.filterwarnings('ignore')

from load_data import *
from ms_2bias3_newtest6 import *
from utils import *
import torch.nn as nn
from torch_sparse import SparseTensor
from auto_profile import infer_profile, apply_profile_to_args



def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed_num', type=int, default=5, help='The number of random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='loan',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'credit', 'german'])
    parser.add_argument('--gate_mode', type=str, default='auto',
                        choices=['auto', 'fair', 'fairmoe_full'],
                        help='gate mode selection: auto(fairmoe_full) or force fair/fairmoe_full')
    parser.add_argument('--acc_backbone', type=str, default='auto',
                        choices=['auto', 'gcn_gat', 'mlp_gat', 'gcn+gat', 'mlp+gat'],
                        help='accuracy stream backbone: auto(bail->mlp_gat, others->gcn_gat)')
    parser.add_argument('--acc_combo', type=str, default='auto',
                        choices=['auto', 'gat_gcn', 'gat_mlp', 'gcn_gat', 'mlp_gat',
                                 'gat+gcn', 'gat+mlp', 'gcn+gat', 'mlp+gat'],
                        help='optional accuracy combo override: auto(dataset-aware) or force gat_gcn/gat_mlp')
    parser.add_argument("--channels", type=int, default=4, help="number of channels")
    parser.add_argument('--model', type=str, default='fairmoe', choices=['fairmoe','gcn', 'sage', 'gin', 'jk', 'infomax', 'ssf',
                                                                     'RobustGCN', 'mlpgcn', 'gcnori', 'disengnn',
                                                                     'disengcn', 'pcagcn', 'adagcn', 'adagcn_new'])
    # alpha，控制特征解耦损失权重
    parser.add_argument('--alpha', type=float, default=0.25, help='weight coefficient for disentanglement loss')
    parser.add_argument('--beta', type=float, default=0.25, help='weight coefficient for channel masker loss')

    # 数据特征自适应配置
    parser.add_argument('--auto_profile', action='store_true', default=False,
                        help='auto-select hyper-parameter profile from data signature')
    parser.add_argument('--force', dest='force', type=str, default='auto',
                        choices=['auto', 'balanced', 'lowproxy', 'highproxy', 'nba'],
                        help='force profile when auto_profile is enabled: auto|balanced|lowproxy|highproxy|nba')
    parser.add_argument('--save_results', type=bool, default=False)

    args = parser.parse_known_args()[0]

    # 显式参数跟踪：实现 CLI 显式值 > auto profile > 默认值
    explicit_flags = {
        'lr': '--lr' in sys.argv,
        'weight_decay': '--weight_decay' in sys.argv,
        'dropout': '--dropout' in sys.argv,
        'alpha': '--alpha' in sys.argv,
        'beta': '--beta' in sys.argv,
        'hidden': '--hidden' in sys.argv,
        'acc_combo': '--acc_combo' in sys.argv,
    }
    args.explicit_flags = explicit_flags

    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    args.device = torch.device('cuda' if args.cuda else 'cpu')

    return args

def set_seed(args, seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 关闭CUDA非确定性算法
    torch.backends.cudnn.deterministic = True  # 强制确定性算法
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def run(args, seed_idx=None):
    torch.set_printoptions(threshold=float('inf'))
    """
    Load data
    """
    fair_dataset = FairDataset(args.dataset, args.device)
    fair_dataset.load_data()

    if args.auto_profile and not getattr(args, '_auto_profile_applied', False):
        force_alias_map = {
            'balanced': 'profile_dense_balanced',
            'lowproxy': 'profile_sparse_lowproxy',
            'highproxy': 'profile_sparse_highimbalance',
            'nba': 'profile_nba_hidden16',
        }
        forced_profile_name = None if args.force == 'auto' else force_alias_map[args.force]
        profile = infer_profile(fair_dataset, force_profile_name=forced_profile_name)
        apply_profile_to_args(args, profile, args.explicit_flags)
        print(f"auto_profile: selected={profile['name']}")
        args._auto_profile_applied = True

    num_class = 1
    args.nfeat = fair_dataset.features.shape[1]
    args.nnode = fair_dataset.features.shape[0]
    args.nclass = num_class

    """
    Build model
    """
    fairmoe_trainer = FairMoE(
        args,
        gate_mode=gate_mode,
        min_fair=getattr(args, 'min_fair', 0.26),
        warmup_steps=getattr(args, 'warmup_steps', 300),
        gate_fair_weight=getattr(args, 'gate_fair_weight', 2.4),
        res_val_f1_weight=getattr(args, 'res_val_f1_weight', 0.5),
        res_val_roc_weight=getattr(args, 'res_val_roc_weight', 0.5),
        min_auc=getattr(args, 'min_auc', None),
        min_f1=getattr(args, 'min_f1', None),
        max_parity=getattr(args, 'max_parity', None),
        max_equality=getattr(args, 'max_equality', None),
        constraint_mode=getattr(args, 'constraint_mode', 'soft'),
        constraint_penalty_weight=getattr(args, 'constraint_penalty_weight', 5.0),
        use_group_fair_loss=getattr(args, 'use_group_fair_loss', False),
        lambda_dp=getattr(args, 'lambda_dp', 0.0),
        lambda_eo=getattr(args, 'lambda_eo', 0.0),
        gate_fair_warmup_steps=getattr(args, 'gate_fair_warmup_steps', 0),
        group_fair_warmup_steps=getattr(args, 'group_fair_warmup_steps', 0),
        sel_two=getattr(args, 'sel_two', False),
        sel_tol=getattr(args, 'sel_tol', 0.5),
        grp_on=getattr(args, 'grp_on', False),
        grp_pow=getattr(args, 'grp_pow', 1.0),
        mix_pref=getattr(args, 'mix_pref', False),
        fair_cap=getattr(args, 'fair_cap', 2.0),
        gain_min=getattr(args, 'gain_min', 0.0),
        ref_on=getattr(args, 'ref_on', False),
        ref_ep=getattr(args, 'ref_ep', 150),
        ref_lr=getattr(args, 'ref_lr', 0.3),
        ref_fm=getattr(args, 'ref_fm', 2.0),
        ref_tol=getattr(args, 'ref_tol', 0.5),
        stab_on=getattr(args, 'stab_on', False),
        stab_ep=getattr(args, 'stab_ep', 200),
        stab_w=getattr(args, 'stab_w', 3.8),
        gate_tar=getattr(args, 'gate_tar', 0.85),
        gate_mw=getattr(args, 'gate_mw', 10.0),
        gate_vw=getattr(args, 'gate_vw', 1.0),
        gate_hi=getattr(args, 'gate_hi', 0.85),
        gate_hw=getattr(args, 'gate_hw', 5.0),
        gate_ha=getattr(args, 'gate_ha', 0.8),
    )

    desc = f"Seed {seed_idx}" if seed_idx is not None else "Seed"
    args.pbar = tqdm(total=args.epochs, desc=desc, unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")

    """
    Train model
    """
    auc_roc_test, f1_s_test, acc_test, parity_test, equality_test = fairmoe_trainer.train_fit(
        fair_dataset,
        args.epochs,
        alpha=args.alpha,  # 解耦损失权重
        beta=args.beta,  # Mask损失权重
        pbar=args.pbar,
    )

    return auc_roc_test, f1_s_test, acc_test, parity_test, equality_test


if __name__ == '__main__':
    args = args_parser()

    print("✅ Start training...")
    auto_gate_mode = "fairmoe_full"

    if args.gate_mode == 'auto':
        gate_mode = auto_gate_mode
        gate_mode_src = 'auto'
    else:
        gate_mode = args.gate_mode
        gate_mode_src = 'forced'

    print(
            f"dataset={args.dataset} use "
            f"gate_mode={gate_mode} ({gate_mode_src})"
        )

    combo_alias = {
        'auto': 'auto',
        'gat_gcn': 'gcn_gat',
        'gat+gcn': 'gcn_gat',
        'gcn_gat': 'gcn_gat',
        'gcn+gat': 'gcn_gat',
        'gat_mlp': 'mlp_gat',
        'gat+mlp': 'mlp_gat',
        'mlp_gat': 'mlp_gat',
        'mlp+gat': 'mlp_gat',
    }

    if args.acc_combo != 'auto':
        resolved_acc_backbone = combo_alias.get(str(args.acc_combo).lower(), 'auto')
        acc_backbone_src = 'forced(acc_combo)'
    elif args.acc_backbone == 'auto':
        resolved_acc_backbone = 'mlp_gat' if args.dataset == 'bail' else 'gcn_gat'
        acc_backbone_src = 'auto'
    else:
        resolved_acc_backbone = combo_alias.get(str(args.acc_backbone).lower(), 'auto')
        acc_backbone_src = 'forced(acc_backbone)'
    print(f"dataset={args.dataset} use acc_backbone={resolved_acc_backbone} ({acc_backbone_src})")

    model_num = 1
    results = Results(args.seed_num, model_num, args)

    for seed in range(args.seed_num):
        set_seed(args, seed)

        results.auc[seed, :], results.f1[seed, :], results.acc[seed, :], results.parity[seed, :], \
            results.equality[seed, :] = run(args, seed_idx=seed + 1)

    print("🎉 Training finished.")
    # reporting results
    results.report_results()
    if args.save_results:
        results.save_results(args)
