import torch.nn as nn
import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from utils import *
from torch_geometric.nn import MessagePassing, GCNConv
from torch_sparse import SparseTensor, matmul
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.metrics import confusion_matrix


# # === 自动寻找最佳阈值 ===
from sklearn.metrics import f1_score
import numpy as np


THRESH_GRID_F1 = np.arange(0.3, 0.8, 0.01)
THRESH_GRID_CONS = np.arange(0.1, 0.9, 0.01)


def get_best_threshold(y_true, y_probs):
    """
    寻找能使 F1-score 最大化的最佳阈值
    """
    best_f1 = -1
    best_thresh = 0.5

    for thresh in THRESH_GRID_F1:
        # 预测类别
        y_pred = (y_probs >= thresh).astype(int)

        score = f1_score(y_true, y_pred, average='binary')

        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    return best_thresh


def _to_ratio(v):
    """支持 0~1 或 0~100 输入，>1 视作百分数。"""
    if v is None:
        return None
    v = float(v)
    return v / 100.0 if v > 1.0 else v


def _to_auc_tol_ratio(v):
    """
    AUC 容差输入支持两种形式：
    - 百分点：0.3 / 0.5（常用）-> 转成 0.003 / 0.005
    - 比例：0.003 / 0.005 -> 保持不变
    """
    if v is None:
        return 0.005
    v = float(v)
    if v <= 0.05:
        return v
    return v / 100.0


def compute_constraint_violation(
    auc,
    f1,
    parity,
    equality,
    min_auc=None,
    min_f1=None,
    max_parity=None,
    max_equality=None,
):
    total = 0.0
    if min_auc is not None:
        total += max(0.0, min_auc - auc)
    if min_f1 is not None:
        total += max(0.0, min_f1 - f1)
    if max_parity is not None:
        total += max(0.0, parity - max_parity)
    if max_equality is not None:
        total += max(0.0, equality - max_equality)
    return total


def get_best_threshold_constrained(
    y_true,
    y_probs,
    sens,
    min_f1=None,
    max_parity=None,
    max_equality=None,
):
    """
    先找满足公平约束的阈值里 F1 最高者；
    若无可行阈值，返回违规最小（并尽量高 F1）的阈值。
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_probs = np.asarray(y_probs).reshape(-1)
    sens = np.asarray(sens).reshape(-1)

    best_th = 0.5
    best_f1 = -1.0

    best_relaxed_th = 0.5
    best_relaxed_violation = float("inf")
    best_relaxed_f1 = -1.0

    for th in THRESH_GRID_CONS:
        pred = (y_probs >= th).astype(int)
        f1_cur = f1_score(y_true, pred, average='binary')
        p_cur, e_cur = fair_metric(pred, y_true, sens)
        violation = compute_constraint_violation(
            auc=0.0,
            f1=f1_cur,
            parity=p_cur,
            equality=e_cur,
            min_auc=None,
            min_f1=min_f1,
            max_parity=max_parity,
            max_equality=max_equality,
        )

        if violation == 0.0 and f1_cur > best_f1:
            best_f1 = f1_cur
            best_th = th

        if (violation < best_relaxed_violation) or (
            violation == best_relaxed_violation and f1_cur > best_relaxed_f1
        ):
            best_relaxed_violation = violation
            best_relaxed_f1 = f1_cur
            best_relaxed_th = th

    if best_f1 >= 0.0:
        return best_th
    return best_relaxed_th


def compute_group_fair_losses_from_probs(probs, labels, sens):
    """
    可微分组公平损失：
    - DP: |E[p|s=0] - E[p|s=1]|
    - EO: |E[p|s=0,y=1] - E[p|s=1,y=1]|
    """
    probs = probs.view(-1)
    labels = labels.view(-1)
    sens = sens.view(-1)

    idx_s0 = sens < 0.5
    idx_s1 = sens >= 0.5
    zero = probs.new_tensor(0.0)

    if idx_s0.any() and idx_s1.any():
        loss_dp = torch.abs(probs[idx_s0].mean() - probs[idx_s1].mean())
    else:
        loss_dp = zero

    idx_y1 = labels >= 0.5
    idx_s0_y1 = idx_s0 & idx_y1
    idx_s1_y1 = idx_s1 & idx_y1
    if idx_s0_y1.any() and idx_s1_y1.any():
        loss_eo = torch.abs(probs[idx_s0_y1].mean() - probs[idx_s1_y1].mean())
    else:
        loss_eo = zero

    return loss_dp, loss_eo


def build_group_reweight(labels, sens, idx_train, power=1.0):
    """
    仅训练集上的 (sens, label) 四组逆频率重加权：
    w_g = (N_train / (4 * N_g)) ** power
    """
    weights = torch.ones_like(labels, dtype=torch.float32)
    idx = idx_train.view(-1)
    idx_dev = idx.to(labels.device)
    y = labels[idx_dev].view(-1).long()
    s = sens[idx_dev].view(-1).long()
    gid = s * 2 + y  # {0,1,2,3}

    n_total = float(idx.numel())
    n_groups = 4.0
    eps = 1e-12

    for g in range(4):
        g_mask_local = (gid == g)
        n_g = float(g_mask_local.sum().item())
        if n_g < 1.0:
            continue
        w_g = (n_total / (n_groups * (n_g + eps))) ** float(power)
        weights[idx_dev[g_mask_local]] = float(w_g)

    # 归一化到训练集均值约为 1，避免影响整体学习率尺度
    train_mean = weights[idx_dev].mean().clamp_min(1e-12)
    weights = weights / train_mean
    return weights

# def get_best_threshold(y_true, y_probs):
#     """
#     使用 G-Mean (Geometric Mean) 寻找最佳阈值。
#     这是处理类别不平衡较稳健的方法。
#     """
#     best_score = -1
#     best_thresh = 0.5
#
#     # 统一成 1D，避免 (N,1) 与 (N,) 广播成 (N,N) 导致阈值搜索失真
#     y_probs = np.asarray(y_probs).reshape(-1)
#
#     for thresh in np.arange(0.1, 0.9, 0.005):
#         y_pred = (y_probs >= thresh).astype(int)
#
#         # 兼容 Tensor 和 Numpy
#         if isinstance(y_true, torch.Tensor):
#             y_true_np = y_true.cpu().numpy()
#         else:
#             y_true_np = y_true
#         y_true_np = np.asarray(y_true_np).reshape(-1)
#
#         tp = ((y_pred == 1) & (y_true_np == 1)).sum()
#         tn = ((y_pred == 0) & (y_true_np == 0)).sum()
#         fp = ((y_pred == 1) & (y_true_np == 0)).sum()
#         fn = ((y_pred == 0) & (y_true_np == 1)).sum()
#
#         tpr = tp / (tp + fn + 1e-8)
#         tnr = tn / (tn + fp + 1e-8)
#         gmean = np.sqrt(tpr * tnr)
#
#         if gmean > best_score:
#             best_score = gmean
#             best_thresh = thresh
#
#     return best_thresh


class FairMoE(torch.nn.Module):
    def __init__(
        self,
        train_args,
        gate_mode="fairmoe_full",
        min_fair=None,
        warmup_steps=None,
        gate_fair_weight=None,
        res_val_f1_weight=None,
        res_val_roc_weight=None,
        min_auc=None,
        min_f1=None,
        max_parity=None,
        max_equality=None,
        constraint_mode=None,
        constraint_penalty_weight=None,
        use_group_fair_loss=None,
        lambda_dp=None,
        lambda_eo=None,
        gate_fair_warmup_steps=None,
        group_fair_warmup_steps=None,
        sel_two=None,
        sel_tol=None,
        grp_on=None,
        grp_pow=None,
        valgb_on=None,
        valgb_alpha=None,
        mix_pref=None,
        fair_cap=None,
        gain_min=None,
        ref_on=None,
        ref_ep=None,
        ref_lr=None,
        ref_fm=None,
        ref_tol=None,
        stab_on=None,
        stab_ep=None,
        stab_w=None,
        gate_tar=None,
        gate_mw=None,
        gate_vw=None,
        gate_hi=None,
        gate_hw=None,
        gate_ha=None,
    ):
        super(FairMoE, self).__init__()
        self.args = train_args
        self.gate_mode = gate_mode

        # 可调超参数（默认保持当前文件行为）
        self.min_fair = 0.26 if min_fair is None else min_fair
        self.warmup_steps = 300 if warmup_steps is None else warmup_steps
        self.gate_fair_weight = 2.4 if gate_fair_weight is None else gate_fair_weight
        self.res_val_f1_weight = 0.5 if res_val_f1_weight is None else res_val_f1_weight
        self.res_val_roc_weight = 0.5 if res_val_roc_weight is None else res_val_roc_weight

        # 约束式选模（默认关闭，不影响 German 现有结果）
        self.min_auc = _to_ratio(getattr(train_args, 'min_auc', None) if min_auc is None else min_auc)
        self.min_f1 = _to_ratio(getattr(train_args, 'min_f1', None) if min_f1 is None else min_f1)
        self.max_parity = _to_ratio(getattr(train_args, 'max_parity', None) if max_parity is None else max_parity)
        self.max_equality = _to_ratio(getattr(train_args, 'max_equality', None) if max_equality is None else max_equality)
        self.constraint_mode = (
            getattr(train_args, 'constraint_mode', 'soft')
            if constraint_mode is None else constraint_mode
        )
        self.constraint_penalty_weight = (
            getattr(train_args, 'constraint_penalty_weight', 5.0)
            if constraint_penalty_weight is None else constraint_penalty_weight
        )
        self.enable_constraints = any(v is not None for v in [
            self.min_auc, self.min_f1, self.max_parity, self.max_equality
        ])

        # Gate 端可微 DP/EO（默认关闭）
        self.use_group_fair_loss = (
            getattr(train_args, 'use_group_fair_loss', False)
            if use_group_fair_loss is None else use_group_fair_loss
        )
        self.lambda_dp = (
            getattr(train_args, 'lambda_dp', 0.0)
            if lambda_dp is None else lambda_dp
        )
        self.lambda_eo = (
            getattr(train_args, 'lambda_eo', 0.0)
            if lambda_eo is None else lambda_eo
        )
        self.gate_fair_warmup_steps = (
            getattr(train_args, 'gate_fair_warmup_steps', 0)
            if gate_fair_warmup_steps is None else gate_fair_warmup_steps
        )
        self.group_fair_warmup_steps = (
            getattr(train_args, 'group_fair_warmup_steps', 0)
            if group_fair_warmup_steps is None else group_fair_warmup_steps
        )

        self.sel_two = (
            getattr(train_args, 'sel_two', False)
            if sel_two is None else sel_two
        )
        self.sel_tol = _to_auc_tol_ratio(
            getattr(train_args, 'sel_tol', 0.5)
            if sel_tol is None else sel_tol
        )
        self.grp_on = (
            getattr(train_args, 'grp_on', False)
            if grp_on is None else grp_on
        )
        self.grp_pow = (
            getattr(train_args, 'grp_pow', 1.0)
            if grp_pow is None else grp_pow
        )
        self.valgb_on = (
            getattr(train_args, 'valgb_on', False)
            if valgb_on is None else valgb_on
        )
        self.valgb_alpha = (
            getattr(train_args, 'valgb_alpha', 2.0)
            if valgb_alpha is None else valgb_alpha
        )
        self.mix_pref = (
            getattr(train_args, 'mix_pref', False)
            if mix_pref is None else mix_pref
        )
        self.fair_cap = _to_ratio(
            getattr(train_args, 'fair_cap', 2.0)
            if fair_cap is None else fair_cap
        )
        self.gain_min = _to_ratio(
            getattr(train_args, 'gain_min', 0.0)
            if gain_min is None else gain_min
        )

        self.ref_on = (
            getattr(train_args, 'ref_on', False)
            if ref_on is None else ref_on
        )
        self.ref_ep = (
            getattr(train_args, 'ref_ep', 150)
            if ref_ep is None else ref_ep
        )
        self.ref_lr = (
            getattr(train_args, 'ref_lr', 0.3)
            if ref_lr is None else ref_lr
        )
        self.ref_fm = (
            getattr(train_args, 'ref_fm', 2.0)
            if ref_fm is None else ref_fm
        )
        self.ref_tol = _to_auc_tol_ratio(
            getattr(train_args, 'ref_tol', 0.5)
            if ref_tol is None else ref_tol
        )

        self.stab_on = (
            getattr(train_args, 'stab_on', False)
            if stab_on is None else stab_on
        )
        self.stab_ep = (
            getattr(train_args, 'stab_ep', 200)
            if stab_ep is None else stab_ep
        )
        self.stab_w = (
            getattr(train_args, 'stab_w', 3.8)
            if stab_w is None else stab_w
        )
        self.gate_tar = (
            getattr(train_args, 'gate_tar', 0.85)
            if gate_tar is None else gate_tar
        )
        self.gate_mw = (
            getattr(train_args, 'gate_mw', 10.0)
            if gate_mw is None else gate_mw
        )
        self.gate_vw = (
            getattr(train_args, 'gate_vw', 1.0)
            if gate_vw is None else gate_vw
        )
        self.gate_hi = (
            getattr(train_args, 'gate_hi', 0.85)
            if gate_hi is None else gate_hi
        )
        self.gate_hw = (
            getattr(train_args, 'gate_hw', 5.0)
            if gate_hw is None else gate_hw
        )
        self.gate_ha = (
            getattr(train_args, 'gate_ha', 0.8)
            if gate_ha is None else gate_ha
        )

        self.acc_backbone = self._resolve_acc_backbone(
            getattr(train_args, 'acc_backbone', 'auto'),
            getattr(train_args, 'dataset', ''),
            getattr(train_args, 'acc_combo', 'auto'),
        )

        # === 1. 公平流编码器 (Fair Stream) ===
        # === 公平流：多尺度 DisGCN（每个尺度一个）===
        self.encoder_fair = nn.ModuleList([
            DisGCN(
                nfeat=self.args.nfeat,
                nhid=self.args.hidden,
                nclass=self.args.nclass,
                chan_num=4,
                layer_num=2,
                dropout=self.args.dropout
            ).to(self.args.device)
            for _ in range(2)  # 2 个尺度：adj_1, adj_2
        ])

        self.fair_fusion = nn.Linear(
            self.args.hidden * 2,
            self.args.hidden
        ).to(self.args.device)

        # === 2. 准确流编码器 (Acc Stream) ===
        # 将 hidden 维度一分为二，一半给 GAT，一半给 GCN
        half_hidden = self.args.hidden // 2

        self.encoder_bias_gat = nn.ModuleList([
            GATConv(self.args.nfeat, half_hidden // 4, heads=4),
            GATConv(half_hidden, half_hidden // 4, heads=4)
        ]).to(self.args.device)

        self.encoder_bias_mlp = None
        self.encoder_bias_gcn = None
        if self.acc_backbone == 'mlp_gat':
            self.encoder_bias_mlp = nn.ModuleList([
                nn.Linear(self.args.nfeat, half_hidden),
                nn.Linear(half_hidden, half_hidden)
            ]).to(self.args.device)
        else:
            self.encoder_bias_gcn = nn.ModuleList([
                GCNConv(self.args.nfeat, half_hidden),
                GCNConv(half_hidden, half_hidden)
            ]).to(self.args.device)

        # === 3. 专家组件 ===
        self.masker = channel_masker(train_args.hidden).to(self.args.device)

        # 两个独立的分类器
        self.classifier_fair = nn.Linear(train_args.hidden, train_args.nclass).to(self.args.device)
        self.classifier_acc = nn.Linear(train_args.hidden, train_args.nclass).to(self.args.device)

        # 自动门控参数
        self.gate_net = nn.Linear(train_args.hidden, 1).to(self.args.device)

        # === 5. 辅助组件 ===
        self.per_channel_dim = train_args.hidden // train_args.channels
        self.channel_cls = nn.Linear(self.per_channel_dim, train_args.channels).to(self.args.device)

        self.weight1 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.weight2 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        safe_lr = self.args.lr * 0.2
        # === 6. 优化器 ===
        # === Fair Stream Optimizer (baseline-aligned) ===
        self.opt_fair = torch.optim.Adam(
            list(self.encoder_fair.parameters()) +
            list(self.masker.parameters()) +
            list(self.classifier_fair.parameters())
            + [self.weight1] + [self.weight2],#
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

        # === Bias Stream Optimizer (accuracy expert) ===
        self.opt_bias = torch.optim.Adam(
            self._acc_trainable_params(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

        # === Gate Optimizer (ONLY gate) ===
        self.opt_gate = torch.optim.Adam(
            list(self.gate_net.parameters()),
            lr=self.args.lr * 0.1
        )

        # === Channel classifier ===
        self.optimizer_c = torch.optim.Adam(
            list(self.channel_cls.parameters()),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

        # === 7. 损失函数 ===
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_dc = DistCor()
        self.criterion_mul_cls = nn.CrossEntropyLoss()
        self.criterion_mask = FeatCov()

        # 初始化参数
        for m in self.modules():
            self.weights_init(m)

        for enc in self.encoder_fair:
            enc.init_parameters()
            enc.init_edge_weight()

        for m in self.encoder_bias_gat:
            if hasattr(m, 'reset_parameters'): m.reset_parameters()
        if self.encoder_bias_mlp is not None:
            for m in self.encoder_bias_mlp:
                if hasattr(m, 'reset_parameters'): m.reset_parameters()
        if self.encoder_bias_gcn is not None:
            for m in self.encoder_bias_gcn:
                if hasattr(m, 'reset_parameters'): m.reset_parameters()

    @staticmethod
    def _resolve_acc_backbone(acc_backbone, dataset_name, acc_combo='auto'):
        combo_mode = str(acc_combo).lower()
        alias = {
            'gcn+gat': 'gcn_gat',
            'gat+gcn': 'gcn_gat',
            'gcn_gat': 'gcn_gat',
            'gat_gcn': 'gcn_gat',
            'mlp+gat': 'mlp_gat',
            'gat+mlp': 'mlp_gat',
            'mlp_gat': 'mlp_gat',
            'gat_mlp': 'mlp_gat',
            'auto': 'auto',
        }

        combo_mode = alias.get(combo_mode, 'auto')
        if combo_mode != 'auto':
            return combo_mode

        mode = str(acc_backbone).lower()
        mode = alias.get(mode, 'auto')
        if mode == 'auto':
            ds = str(dataset_name).lower()
            return 'mlp_gat' if ds == 'bail' else 'gcn_gat'
        return mode

    def _acc_modules(self):
        modules = [self.encoder_bias_gat]
        if self.acc_backbone == 'mlp_gat':
            modules.append(self.encoder_bias_mlp)
        else:
            modules.append(self.encoder_bias_gcn)
        return modules

    def _acc_trainable_params(self):
        params = []
        for module in self._acc_modules():
            params += list(module.parameters())
        params += list(self.classifier_acc.parameters())
        return params

    def _set_acc_requires_grad(self, requires_grad):
        for module in self._acc_modules():
            for p in module.parameters():
                p.requires_grad = requires_grad
        for p in self.classifier_acc.parameters():
            p.requires_grad = requires_grad

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, adj_1, adj_2, edge_index):
        # === 分支 1：公平 Fair Path（多尺度）===
        h_scale1 = self.encoder_fair[0](x, adj_1)
        h_scale2 = self.encoder_fair[1](x, adj_2)

        # # 多尺度拼接
        # h_multi = torch.cat([h_scale1, h_scale2], dim=1)
        # # 融合回 hidden 维度
        # h_fair = self.fair_fusion(h_multi)

        # h_fair = h_scale1  # 不用 h_scale2
        #
        # h_masked = self.masker(h_fair)
        # out_fair = self.classifier_fair(h_masked)

        #============== 自学习融合多尺度 ===================================
        # 可学习权重融合（权重和为1，增强稳定性）
        weight1_normalized = torch.sigmoid(self.weight1)
        # weight2_normalized = 1 - weight1_normalized
        weight2_normalized = torch.sigmoid(self.weight2) #❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗
        h_fair = weight1_normalized * h_scale1 + weight2_normalized * h_scale2
        h_masked = self.masker(h_fair)
        out_fair = self.classifier_fair(h_masked)
        #=================================================================

        # === 分支 2：准确Bias Path (双尺度 GAT + GCN) ===
        # --- GAT ---
        h_gat = x
        for i, conv in enumerate(self.encoder_bias_gat):
            h_gat = conv(h_gat, edge_index)
            if i < len(self.encoder_bias_gat) - 1:
                h_gat = F.elu(h_gat)
                h_gat = F.dropout(h_gat, p=self.args.dropout, training=self.training)

        if self.acc_backbone == 'mlp_gat':
            h_aux = x
            for i, layer in enumerate(self.encoder_bias_mlp):
                h_aux = layer(h_aux)
                if i < len(self.encoder_bias_mlp) - 1:
                    h_aux = F.relu(h_aux)
                    h_aux = F.dropout(h_aux, p=self.args.dropout, training=self.training)
        else:
            h_aux = x
            for i, conv in enumerate(self.encoder_bias_gcn):
                h_aux = conv(h_aux, edge_index)
                if i < len(self.encoder_bias_gcn) - 1:
                    h_aux = F.relu(h_aux)
                    h_aux = F.dropout(h_aux, p=self.args.dropout, training=self.training)

        # --- 融合 ---
        h_bias = torch.cat([h_gat, h_aux], dim=1)
        out_acc = self.classifier_acc(h_bias)

        # === 融合 Gate ===
        if self.gate_mode == "fair":
            w = torch.ones(
                out_fair.size(0), 1,
                device=out_fair.device
            )
            output = out_fair

        elif self.gate_mode == "fairmoe_full":
            # 自适应 gate 使用
            gate_input = h_bias.detach()
            gate_logit = self.gate_net(gate_input)
            w = torch.sigmoid(gate_logit)
            w = torch.clamp(w, min=self.min_fair)
            output = w * out_fair + (1 - w) * out_acc

        return output, h_fair, h_masked, out_fair, out_acc, w, h_bias

    def train_fit(self, data, epochs, **kwargs):
        loss_w_alpha = kwargs.get('alpha', self.args.alpha)
        loss_w_beta = kwargs.get('beta', self.args.beta)
        pbar = kwargs.get('pbar', None)

        best_res_val = -float('inf')
        roc_test = f1_test = acc_test = parity_test = equality_test = 0.0
        best_epoch = 0
        roc_val = 0.0
        best_val_roc = -float('inf')
        best_state = None

        idx_train_dev = data.idx_train.to(data.features.device)
        idx_val_dev = data.idx_val.to(data.features.device)
        idx_test_dev = data.idx_test.to(data.features.device)

        y_train = data.labels[idx_train_dev].unsqueeze(1).float()
        s_train = data.sens[idx_train_dev].unsqueeze(1).float()
        sens_train_raw = data.sens[idx_train_dev]

        labels_val_np = data.labels[idx_val_dev].cpu().numpy().reshape(-1)
        sens_val_np = data.sens[idx_val_dev].cpu().numpy().reshape(-1)
        labels_test_np = data.labels[idx_test_dev].cpu().numpy().reshape(-1)
        sens_test_np = data.sens[idx_test_dev].cpu().numpy().reshape(-1)

        fair_clip_params = (
            list(self.encoder_fair.parameters()) +
            list(self.masker.parameters()) +
            list(self.classifier_fair.parameters())
        )
        acc_trainable_params = self._acc_trainable_params()
        refine_clip_params = fair_clip_params + list(self.gate_net.parameters())

        channel_slices = [
            slice(i * self.per_channel_dim, (i + 1) * self.per_channel_dim)
            for i in range(self.args.channels)
        ]
        channel_targets = [
            torch.full(
                (data.features.size(0),),
                i,
                device=self.args.device,
                dtype=torch.long,
            )
            for i in range(self.args.channels)
        ]

        mode_on = any([
            self.sel_two,
            self.grp_on,
            self.valgb_on,
            self.mix_pref,
            self.ref_on,
            self.stab_on,
        ])
        use_reweight = mode_on and self.grp_on
        if use_reweight:
            train_sample_w = build_group_reweight(
                labels=data.labels,
                sens=data.sens,
                idx_train=data.idx_train,
                power=self.grp_pow,
            ).to(self.args.device)
            idx_train_w = idx_train_dev.to(train_sample_w.device)
        else:
            train_sample_w = None
            idx_train_w = None

        # 两阶段选模状态：先 AUC 窗口，再公平最小
        sel_best_auc_seen = -float('inf')
        sel_best_fair_sum = float('inf')
        sel_best_auc_tiebreak = -float('inf')
        sel_selected_val_auc = -float('inf')

        for epoch in range(epochs):
            self.train()

            # # ==========================================
            # # 🔥 预热策略 (Warm-up Strategy)
            # # 前 30 个 epoch 只专心学分类,对小数据集友好
            # # ==========================================
            # if epoch < 30:
            #     actual_beta = 0.0  # 暂时关闭 Mask惩罚
            #     actual_alpha = 0.0  # 暂时关闭解耦惩罚
            # else:
            #     actual_beta = loss_w_beta
            #     actual_alpha = loss_w_alpha

            # ==========================================
            # 🔥 修改：线性预热 (Linear Warm-up)
            # 不要突然袭击，要在前 200 轮慢慢加上去
            # ==========================================
            if self.gate_mode == "fairmoe_full" and epoch < self.warmup_steps:
                # 系数从 0.0 慢慢涨到 1.0
                factor = epoch / self.warmup_steps
                actual_beta = loss_w_beta * factor
                actual_alpha = loss_w_alpha * factor
            else:
                # 预热结束后保持满额
                actual_beta = loss_w_beta
                actual_alpha = loss_w_alpha

            # ======== Forward ========
            output, h_fair, h_masked, out_fair, out_acc, w, h_bias = \
                self(
                    data.features,
                    data.adj_1,
                    data.adj_2,
                    data.edge_index
                )

            # ======== Compute losses (shared parts) ========
            fair_logits_train = out_fair[idx_train_dev]
            acc_logits_train = out_acc[idx_train_dev]
            if use_reweight:
                w_train = train_sample_w[idx_train_w].view(-1, 1)
                loss_fair_cls = (
                    F.binary_cross_entropy_with_logits(
                        fair_logits_train,
                        y_train,
                        reduction='none'
                    ) * w_train
                ).mean()
                loss_acc_cls = (
                    F.binary_cross_entropy_with_logits(
                        acc_logits_train,
                        y_train,
                        reduction='none'
                    ) * w_train
                ).mean()
            else:
                loss_fair_cls = self.criterion_bce(fair_logits_train, y_train)
                loss_acc_cls = self.criterion_bce(acc_logits_train, y_train)

            loss_mask = self.criterion_mask(
                h_masked[idx_train_dev], sens_train_raw
            )

            # channel losses
            loss_chan = 0.0
            loss_disen = 0.0
            for i in range(self.args.channels):
                chan_output = self.channel_cls(
                    h_fair[:, channel_slices[i]]
                )
                loss_chan += self.criterion_mul_cls(chan_output, channel_targets[i])

            len_per_channel = self.per_channel_dim
            for i in range(self.args.channels):
                for j in range(i + 1, self.args.channels):
                    loss_disen += self.criterion_dc(
                        h_fair[idx_train_dev,
                        i * len_per_channel:(i + 1) * len_per_channel],
                        h_fair[idx_train_dev,
                        j * len_per_channel:(j + 1) * len_per_channel]
                    )


            # ======================================================
            # 1️⃣ Fair Stream update (baseline-aligned)
            # ======================================================

            self.opt_fair.zero_grad()
            self.optimizer_c.zero_grad()

            loss_fair = (
                    loss_fair_cls +
                    actual_beta * loss_mask +
                    actual_alpha * (loss_chan + loss_disen)
            )

            loss_fair.backward()
            torch.nn.utils.clip_grad_norm_(
                fair_clip_params,
                1.0
            )

            self.opt_fair.step()
            self.optimizer_c.step()

            if self.gate_mode == "fair":
                skip_bias_and_gate = True
            else:
                skip_bias_and_gate = False

            # legacy credit warmup：先只练公平专家
            if self.stab_on and epoch < self.stab_ep:
                skip_bias_and_gate = True


            if not skip_bias_and_gate:
                # ======================================================
                # 2️⃣ Bias Stream update (accuracy expert)
                # ======================================================
                self.opt_bias.zero_grad()

                loss_acc_cls.backward()
                torch.nn.utils.clip_grad_norm_(
                    acc_trainable_params,
                    1.0
                )

                self.opt_bias.step()

                # ======================================================
                # 3️⃣ Gate update (ONLY gate, experts detached)
                # ======================================================

                # --- 临时冻结 encoder（仅 Gate 更新期间）---
                for p in self.encoder_fair.parameters():
                    p.requires_grad = False
                self._set_acc_requires_grad(False)


                # if not skip_bias_and_gate:
                self.opt_gate.zero_grad()

                # === Gate forward (experts detached) ===
                out_fair_detach = out_fair.detach()
                out_acc_detach = out_acc.detach()

                gate_logit = self.gate_net(h_bias.detach())
                gate_weight = torch.sigmoid(gate_logit)
                gate_weight = torch.clamp(gate_weight, min=self.min_fair)

                output_gate = gate_weight * out_fair_detach + \
                              (1 - gate_weight) * out_acc_detach

                # === 1. Task loss (accuracy) ===
                gate_logits_train = output_gate[idx_train_dev]
                if use_reweight:
                    w_train = train_sample_w[idx_train_w].view(-1, 1)
                    loss_gate_task = (
                        F.binary_cross_entropy_with_logits(
                            gate_logits_train,
                            y_train,
                            reduction='none'
                        ) * w_train
                    ).mean()
                else:
                    loss_gate_task = self.criterion_bce(gate_logits_train, y_train)

                # === 2. Fairness penalty ===
                loss_gate_fair = self.criterion_mask(
                    output_gate[idx_train_dev],
                    s_train
                )

                if self.gate_fair_warmup_steps > 0 and epoch < self.gate_fair_warmup_steps:
                    gate_fair_factor = float(epoch) / float(self.gate_fair_warmup_steps)
                else:
                    gate_fair_factor = 1.0

                loss_gate = loss_gate_task + (self.gate_fair_weight * gate_fair_factor) * loss_gate_fair

                if self.use_group_fair_loss and (self.lambda_dp > 0.0 or self.lambda_eo > 0.0):
                    probs_gate = torch.sigmoid(output_gate[idx_train_dev])
                    loss_dp, loss_eo = compute_group_fair_losses_from_probs(
                        probs_gate,
                        y_train,
                        s_train,
                    )
                    if self.group_fair_warmup_steps > 0 and epoch < self.group_fair_warmup_steps:
                        group_fair_factor = float(epoch) / float(self.group_fair_warmup_steps)
                    else:
                        group_fair_factor = 1.0
                    loss_gate = loss_gate + group_fair_factor * (self.lambda_dp * loss_dp + self.lambda_eo * loss_eo)

                if self.stab_on:
                    alpha_mean = gate_weight.mean()
                    loss_mean_reg = self.gate_mw * torch.abs(alpha_mean - self.gate_tar)
                    loss_var_reg = self.gate_vw * gate_weight.var()
                    loss_gate = loss_gate + loss_mean_reg + loss_var_reg
                    if alpha_mean > self.gate_hi:
                        loss_gate = loss_gate + self.gate_hw * (
                            alpha_mean - self.gate_ha
                        )

                loss_gate.backward()
                self.opt_gate.step()

                # --- 解冻 encoder，供下一个 epoch 使用 ---
                for p in self.encoder_fair.parameters():
                    p.requires_grad = True
                self._set_acc_requires_grad(True)


            # ======================================================
            # Validation
            # ======================================================
            if epoch % 10 == 0:
                self.eval()
                with torch.inference_mode():
                    output_eval, _, _, out_fair_eval, _, _, _ = self(
                        data.features,
                        data.adj_1,
                        data.adj_2,
                        data.edge_index
                    )

                eval_logits = output_eval
                if self.stab_on and epoch < self.stab_ep:
                    eval_logits = out_fair_eval

                # Bail 实验1：优先公平专家，仅在混合输出满足公平预算且准确率有提升时才采纳
                if (
                    mode_on and
                    self.gate_mode == "fairmoe_full" and
                    self.mix_pref and
                    not (self.stab_on and epoch < self.stab_ep)
                ):
                    sens_val_policy_np = sens_val_np

                    probs_fair = torch.sigmoid(out_fair_eval[idx_val_dev]).detach().cpu().numpy().reshape(-1)
                    th_fair = get_best_threshold(labels_val_np, probs_fair)
                    pred_fair = (probs_fair >= th_fair).astype(int).reshape(-1)
                    acc_fair = accuracy_score(labels_val_np, pred_fair)

                    probs_mix = torch.sigmoid(output_eval[idx_val_dev]).detach().cpu().numpy().reshape(-1)
                    th_mix = get_best_threshold(labels_val_np, probs_mix)
                    pred_mix = (probs_mix >= th_mix).astype(int).reshape(-1)
                    acc_mix = accuracy_score(labels_val_np, pred_mix)
                    par_mix, eq_mix = fair_metric(pred_mix, labels_val_np, sens_val_policy_np)

                    mix_fair_ok = (par_mix <= self.fair_cap) and (eq_mix <= self.fair_cap)
                    mix_acc_gain_ok = (acc_mix - acc_fair) >= self.gain_min

                    if mix_fair_ok and mix_acc_gain_ok:
                        eval_logits = output_eval
                    else:
                        eval_logits = out_fair_eval

                probs_val = torch.sigmoid(eval_logits[idx_val_dev]).detach().cpu().numpy().reshape(-1)
                if self.enable_constraints:
                    best_thresh = get_best_threshold_constrained(
                        labels_val_np,
                        probs_val,
                        sens=sens_val_np,
                        min_f1=self.min_f1,
                        max_parity=self.max_parity,
                        max_equality=self.max_equality,
                    )
                else:
                    best_thresh = get_best_threshold(labels_val_np, probs_val)
                # best_thresh = 0.5
                y_pred_val_np = (probs_val >= best_thresh).astype(int).reshape(-1)
                acc_val = accuracy_score(labels_val_np, y_pred_val_np)
                roc_val = roc_auc_score(labels_val_np, probs_val)
                f1_val = f1_score(labels_val_np, y_pred_val_np)
                parity, equality = fair_metric(y_pred_val_np,
                                               labels_val_np,
                                               sens_val_np)


                # ================= Dataset-aware model selection =================
                if self.gate_mode == "fairmoe_full":
                    base_score = self.res_val_f1_weight * f1_val + self.res_val_roc_weight * roc_val
                    if self.valgb_on:
                        res_val = (
                            roc_val + f1_val + acc_val
                            - self.valgb_alpha * (parity + equality)
                        )
                    elif self.stab_on:
                        res_val = roc_val - self.stab_w * parity
                    elif mode_on and self.sel_two:
                        # 两阶段：保留 AUC 在窗口内，再按 parity+equality 最小选
                        sel_best_auc_seen = max(sel_best_auc_seen, roc_val)
                        # 仅在 warmup 后启用两阶段筛选，避免早期低AUC“锁死”
                        if epoch < self.warmup_steps:
                            auc_in_window = False
                        else:
                            auc_in_window = roc_val >= (sel_best_auc_seen - self.sel_tol)
                        fairness_sum = parity + equality
                        if auc_in_window:
                            # 仅作为日志分数，真正比较逻辑在下面的 update 条件
                            res_val = -fairness_sum
                        else:
                            res_val = -float('inf')
                    elif self.enable_constraints:
                        violation = compute_constraint_violation(
                            auc=roc_val,
                            f1=f1_val,
                            parity=parity,
                            equality=equality,
                            min_auc=self.min_auc,
                            min_f1=self.min_f1,
                            max_parity=self.max_parity,
                            max_equality=self.max_equality,
                        )
                        if self.constraint_mode == 'hard' and violation > 0.0:
                            res_val = -float('inf')
                        else:
                            res_val = base_score - self.constraint_penalty_weight * violation
                    else:
                        res_val = base_score
                    # res_val =  f1_val + 1.8 * roc_val❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗❗
                else: #“fair”模式下
                    res_val = acc_val + roc_val - parity - equality


                should_update = (res_val > best_res_val)
                if mode_on and self.sel_two and self.gate_mode == "fairmoe_full":
                    fairness_sum = parity + equality
                    if epoch < self.warmup_steps:
                        auc_in_window = False
                    else:
                        auc_in_window = roc_val >= (sel_best_auc_seen - self.sel_tol)
                    if auc_in_window:
                        selected_outdated = sel_selected_val_auc < (sel_best_auc_seen - self.sel_tol)
                        if selected_outdated:
                            should_update = True
                        elif fairness_sum < sel_best_fair_sum - 1e-12:
                            should_update = True
                        elif abs(fairness_sum - sel_best_fair_sum) <= 1e-12 and roc_val > sel_best_auc_tiebreak:
                            should_update = True
                        else:
                            should_update = False
                    else:
                        should_update = False

                if should_update:
                    best_res_val = res_val
                    best_epoch = epoch
                    best_val_roc = roc_val
                    best_state = {
                        k: v.detach().clone()
                        for k, v in self.state_dict().items()
                    }
                    if mode_on and self.sel_two and self.gate_mode == "fairmoe_full":
                        sel_best_fair_sum = parity + equality
                        sel_best_auc_tiebreak = roc_val
                        sel_selected_val_auc = roc_val

                    probs_test = torch.sigmoid(
                        eval_logits[idx_test_dev]
                    ).cpu().numpy().reshape(-1)
                    y_pred_test = (probs_test >= best_thresh).astype(int).reshape(-1)

                    roc_test = roc_auc_score(
                        labels_test_np,
                        probs_test
                    )
                    acc_test = accuracy_score(
                        labels_test_np,
                        y_pred_test
                    )
                    f1_test = f1_score(
                        labels_test_np,
                        y_pred_test
                    )

                    parity_test, equality_test = fair_metric(
                        y_pred_test,
                        labels_test_np,
                        sens_test_np
                    )

            if pbar is not None:
                pbar.set_postfix({
                    'loss_Fair': f"{loss_fair.item():.2f}",
                    'AUC':roc_val,
                    # 'Acc': f"{loss_acc_cls.item():.2f}",
                    # 'Gate': f"{loss_gate.item():.2f}",
                    'BestEp': best_epoch
                })
                pbar.update(1)

        if (
            mode_on and
            self.gate_mode == "fairmoe_full" and
            self.ref_on and
            best_state is not None and
            self.ref_ep > 0
        ):
            # 从 stage-1 最优点出发
            self.load_state_dict(best_state)

            # 冻结 accuracy expert
            self._set_acc_requires_grad(False)

            # 缩小学习率
            lr_scale = max(1e-6, float(self.ref_lr))
            opt_lr_backup = {
                'opt_fair': [pg['lr'] for pg in self.opt_fair.param_groups],
                'opt_gate': [pg['lr'] for pg in self.opt_gate.param_groups],
                'opt_c': [pg['lr'] for pg in self.optimizer_c.param_groups],
            }
            for pg in self.opt_fair.param_groups:
                pg['lr'] = pg['lr'] * lr_scale
            for pg in self.opt_gate.param_groups:
                pg['lr'] = pg['lr'] * lr_scale
            for pg in self.optimizer_c.param_groups:
                pg['lr'] = pg['lr'] * lr_scale

            refine_best_state = {
                k: v.detach().clone()
                for k, v in self.state_dict().items()
            }
            refine_best_fair = parity_test + equality_test
            refine_best_score = -float('inf')
            refine_best_val_roc = best_val_roc
            refine_best = (roc_test, f1_test, acc_test, parity_test, equality_test)

            for _ in range(self.ref_ep):
                self.train()
                output, h_fair, h_masked, out_fair, out_acc, w, h_bias = self(
                    data.features,
                    data.adj_1,
                    data.adj_2,
                    data.edge_index
                )

                logits_train = output[idx_train_dev]
                if use_reweight:
                    w_train = train_sample_w[idx_train_w].view(-1, 1)
                    loss_task = (
                        F.binary_cross_entropy_with_logits(
                            logits_train,
                            y_train,
                            reduction='none'
                        ) * w_train
                    ).mean()
                else:
                    loss_task = self.criterion_bce(logits_train, y_train)

                if use_reweight:
                    w_train = train_sample_w[idx_train_w].view(-1, 1)
                    loss_fair_cls = (
                        F.binary_cross_entropy_with_logits(
                            out_fair[idx_train_dev],
                            y_train,
                            reduction='none'
                        ) * w_train
                    ).mean()
                else:
                    loss_fair_cls = self.criterion_bce(out_fair[idx_train_dev], y_train)
                loss_fair_pen = self.criterion_mask(logits_train, s_train)

                group_fair_term = logits_train.new_tensor(0.0)
                if self.use_group_fair_loss and (self.lambda_dp > 0.0 or self.lambda_eo > 0.0):
                    probs_gate = torch.sigmoid(logits_train)
                    loss_dp, loss_eo = compute_group_fair_losses_from_probs(
                        probs_gate,
                        y_train,
                        s_train,
                    )
                    group_fair_term = self.lambda_dp * loss_dp + self.lambda_eo * loss_eo

                loss_refine = loss_task + 0.3 * loss_fair_cls + self.ref_fm * (loss_fair_pen + group_fair_term)

                self.opt_fair.zero_grad()
                self.opt_gate.zero_grad()
                self.optimizer_c.zero_grad()
                loss_refine.backward()
                torch.nn.utils.clip_grad_norm_(
                    refine_clip_params,
                    1.0
                )
                self.opt_fair.step()
                self.opt_gate.step()
                self.optimizer_c.step()

                self.eval()
                with torch.inference_mode():
                    output_eval, _, _, _, _, _, _ = self(
                        data.features,
                        data.adj_1,
                        data.adj_2,
                        data.edge_index
                    )

                probs_val = torch.sigmoid(output_eval[idx_val_dev]).detach().cpu().numpy().reshape(-1)
                if self.enable_constraints:
                    best_thresh = get_best_threshold_constrained(
                        labels_val_np,
                        probs_val,
                        sens=sens_val_np,
                        min_f1=self.min_f1,
                        max_parity=self.max_parity,
                        max_equality=self.max_equality,
                    )
                else:
                    best_thresh = get_best_threshold(labels_val_np, probs_val)

                y_pred_val_np = (probs_val >= best_thresh).astype(int).reshape(-1)
                acc_val_ref = accuracy_score(labels_val_np, y_pred_val_np)
                roc_val_ref = roc_auc_score(labels_val_np, probs_val)
                f1_val_ref = f1_score(labels_val_np, y_pred_val_np)
                parity_ref, equality_ref = fair_metric(y_pred_val_np, labels_val_np, sens_val_np)

                # AUC 保护：不允许跌破 stage-1 最优太多
                fair_sum_ref = parity_ref + equality_ref
                if self.valgb_on:
                    cur_ref_score = (
                        roc_val_ref + f1_val_ref + acc_val_ref
                        - self.valgb_alpha * fair_sum_ref
                    )
                    should_update_ref = cur_ref_score > refine_best_score
                else:
                    if roc_val_ref < (best_val_roc - self.ref_tol):
                        continue
                    cur_ref_score = None
                    should_update_ref = (
                        (fair_sum_ref < refine_best_fair - 1e-12) or
                        (
                            abs(fair_sum_ref - refine_best_fair) <= 1e-12
                            and roc_val_ref > refine_best_val_roc
                        )
                    )

                if should_update_ref:
                    probs_test = torch.sigmoid(output_eval[idx_test_dev]).cpu().numpy().reshape(-1)
                    y_pred_test = (probs_test >= best_thresh).astype(int).reshape(-1)
                    roc_test_ref = roc_auc_score(labels_test_np, probs_test)
                    acc_test_ref = accuracy_score(labels_test_np, y_pred_test)
                    f1_test_ref = f1_score(labels_test_np, y_pred_test)
                    parity_test_ref, equality_test_ref = fair_metric(y_pred_test, labels_test_np, sens_test_np)

                    refine_best_fair = fair_sum_ref
                    if cur_ref_score is not None:
                        refine_best_score = cur_ref_score
                    refine_best_val_roc = roc_val_ref
                    refine_best = (roc_test_ref, f1_test_ref, acc_test_ref, parity_test_ref, equality_test_ref)
                    refine_best_state = {
                        k: v.detach().clone()
                        for k, v in self.state_dict().items()
                    }

            # 载入 refinement 最优
            self.load_state_dict(refine_best_state)
            roc_test, f1_test, acc_test, parity_test, equality_test = refine_best

            # 恢复学习率
            for pg, lr_old in zip(self.opt_fair.param_groups, opt_lr_backup['opt_fair']):
                pg['lr'] = lr_old
            for pg, lr_old in zip(self.opt_gate.param_groups, opt_lr_backup['opt_gate']):
                pg['lr'] = lr_old
            for pg, lr_old in zip(self.optimizer_c.param_groups, opt_lr_backup['opt_c']):
                pg['lr'] = lr_old

            # 解冻 accuracy expert
            self._set_acc_requires_grad(True)

        if pbar is not None:
            pbar.close()

        return roc_test, f1_test, acc_test, parity_test, equality_test


class DisenLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, channels, reduce=True):
        super(DisenLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channels = channels
        self.per_channel_dim = self.out_dim // self.channels
        self.reduce = reduce

        self.lin_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for i in range(channels):
            if reduce:
                self.lin_layers.append(nn.Linear(in_features=in_dim, out_features=self.per_channel_dim))
                self.conv_layers.append(
                    Linear(in_channels=self.per_channel_dim, out_channels=self.per_channel_dim, bias=False,
                           weight_initializer='glorot'))
            else:
                self.conv_layers.append(Linear(in_channels=self.in_dim, out_channels=self.per_channel_dim, bias=False,
                                               weight_initializer='glorot'))
        self.bias_list = nn.ParameterList(
            nn.Parameter(torch.empty(size=(1, self.per_channel_dim), dtype=torch.float), requires_grad=True) for i in
            range(self.channels))

    def get_reddim_k(self, x):
        z_feats = []
        for k in range(self.channels):
            z_feat = self.lin_layers[k](x)
            z_feats.append(z_feat)
        return z_feats

    def get_k_feature(self, x):
        z_feats = []
        for k in range(self.channels):
            z_feats.append(x)
        return z_feats

    def forward(self, x, edge_index, edge_weight):
        assert self.channels == edge_weight.shape[
            1], "axis dimension in direction 1 need to be equal to channels number"
        if self.reduce:
            z_feats = self.get_reddim_k(x)
        else:
            z_feats = self.get_k_feature(x)
        c_feats = []
        for k, layer in enumerate(self.conv_layers):
            c_temp = layer(z_feats[k])
            edge_index_copy = edge_index.clone()
            if not edge_index_copy.has_value():
                edge_index_copy = edge_index_copy.fill_value(1., dtype=None)
            edge_index_copy.storage.set_value_(edge_index_copy.storage.value() * edge_weight[:, k])
            out = self.propagate(edge_index_copy, x=c_temp)
            if self.bias_list is not None:
                out = out + self.bias_list[k]
            c_feats.append(F.normalize(out, p=2, dim=1))
        output = torch.cat(c_feats, dim=1)
        return output

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


class DisGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, chan_num, layer_num, dropout=0.5):
        super(DisGCN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.dropout_rate = dropout
        self.chan_num = chan_num
        self.layer_num = layer_num
        self.edge_weight = None

        self.assigner = NeiborAssigner(nfeat, chan_num)
        self.disenlayers = nn.ModuleList()
        for i in range(layer_num - 1):
            if i == 0:
                self.disenlayers.append(DisenLayer(nfeat, nhid, chan_num))
            else:
                self.disenlayers.append(DisenLayer(nhid, nhid, chan_num))
        self.dropout = nn.Dropout(dropout)
        self._cached_feats_pair = None
        self._cached_pair_key = None

        self.init_parameters()

    def init_parameters(self):
        for i, item in enumerate(self.parameters()):
            torch.nn.init.normal_(item, mean=0, std=1)

    def init_edge_weight(self):
        for m in self.assigner.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        assert isinstance(edge_index, SparseTensor), "Expected input is sparse tensor"
        row = edge_index.storage._row
        col = edge_index.storage._col
        pair_key = (
            x.data_ptr(),
            row.data_ptr(),
            col.data_ptr(),
            x.shape[0],
            x.shape[1],
            str(x.device),
        )
        if self._cached_pair_key != pair_key:
            self._cached_feats_pair = torch.cat([x[col, :], x[row, :]], dim=1)
            self._cached_pair_key = pair_key
        feats_pair = self._cached_feats_pair
        edge_weight = self.assigner(feats_pair.detach())
        for layer in self.disenlayers:
            x = layer(x, edge_index, edge_weight)
            x = self.dropout(x)
        return x


class NeiborAssigner(nn.Module):
    def __init__(self, nfeats, channels):
        super(NeiborAssigner, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=2 * nfeats, out_features=channels),
            nn.Linear(in_features=channels, out_features=channels)
        )

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, features_pair):
        alpha_score = self.layers(features_pair)
        alpha_score = torch.softmax(alpha_score, dim=1)
        return alpha_score


class channel_masker(nn.Module):
    def __init__(self, hid_num):
        super(channel_masker, self).__init__()
        self.hid_num = hid_num
        self.weights = nn.Parameter(torch.distributions.Uniform(0, 1).sample((hid_num, 2)))

    def forward(self, x):
        mask = F.gumbel_softmax(self.weights, tau=0.8, hard=False)[:, 0]
        x = x * mask
        return x
