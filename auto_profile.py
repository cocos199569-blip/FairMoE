import numpy as np


def _safe_rate(x):
    x = float(x)
    if np.isnan(x) or np.isinf(x):
        return 0.0
    return x


def _compute_signature(data):
    y = data.labels.detach().float().view(-1).cpu().numpy()
    s = data.sens.detach().float().view(-1).cpu().numpy()
    x = data.features.detach().float().cpu().numpy()

    n_nodes = int(y.shape[0])
    pos_rate = _safe_rate(np.mean(y))
    sens_rate = _safe_rate(np.mean(s))
    sens_imbalance = _safe_rate(abs(sens_rate - 0.5) * 2.0)

    if np.std(y) > 1e-8 and np.std(s) > 1e-8:
        ys_corr = _safe_rate(abs(np.corrcoef(y, s)[0, 1]))
    else:
        ys_corr = 0.0

    # edge_density: 无向近似密度（做截断，防止异常值）
    e = int(data.edge_index.size(1))
    density = _safe_rate(e / max(1.0, n_nodes * (n_nodes - 1)))
    density = float(np.clip(density, 0.0, 1.0))
    avg_degree = _safe_rate(e / max(1.0, n_nodes))

    # 计算 X 与 sensitive 的相关性统计（用于 profile 相似度）
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    corr_x_s = []
    sensitive_like_cols = 0
    s_std = np.std(s)
    if s_std < 1e-8:
        corr_x_s = [0.0] * x.shape[1]
    else:
        for i in range(x.shape[1]):
            xi = x[:, i]
            # 排除直接等同于敏感属性（或其互补）的特征列，避免把 s 本身当作 proxy
            if np.allclose(xi, s, atol=1e-8) or np.allclose(xi, 1.0 - s, atol=1e-8):
                sensitive_like_cols += 1
                continue
            if np.std(xi) < 1e-8:
                corr_x_s.append(0.0)
            else:
                c = np.corrcoef(xi, s)[0, 1]
                if np.isnan(c) or np.isinf(c):
                    c = 0.0
                corr_x_s.append(abs(c))
    corr_x_s = np.asarray(corr_x_s, dtype=np.float32)
    proxy_ratio = float((corr_x_s > 0.1).mean()) if corr_x_s.size > 0 else 0.0
    max_xs_corr = float(corr_x_s.max()) if corr_x_s.size > 0 else 0.0

    return {
        'dataset': str(getattr(data, 'dataset', '')).lower(),
        'n_nodes': n_nodes,
        'n_nodes_log10': float(np.log10(max(1, n_nodes))),
        'pos_rate': float(pos_rate),
        'sens_rate': float(sens_rate),
        'sens_imbalance': float(sens_imbalance),
        'ys_corr': float(ys_corr),
        'edge_density': float(density),
        'avg_degree': float(avg_degree),
        'proxy_ratio': float(proxy_ratio),
        'max_xs_corr': float(max_xs_corr),
        'sensitive_like_cols': int(sensitive_like_cols),
    }


def _weighted_distance(sig, proto, weights, scales):
    d = 0.0
    for k, v in proto.items():
        s = max(1e-8, float(scales.get(k, 1.0)))
        w = float(weights.get(k, 1.0))
        d += w * abs(float(sig.get(k, 0.0)) - float(v)) / s
    return float(d)


def infer_profile(data, force_profile_name=None):
    sig = _compute_signature(data)

    # ===== 相似度路由：按数据特征 =====
    # 原型取自当前实验的经验中心点，可持续迭代更新
    prototypes = {
        'profile_dense_balanced': {
            'n_nodes_log10': 3.00,
            'pos_rate': 0.70,
            'sens_imbalance': 0.38,
            'ys_corr': 0.08,
            'edge_density': 0.045,
            'avg_degree': 44.0,
            'proxy_ratio': 0.20,
            'max_xs_corr': 0.40,
        },
        'profile_sparse_lowproxy': {
            'n_nodes_log10': 4.25,
            'pos_rate': 0.38,
            'sens_imbalance': 0.05,
            'ys_corr': 0.08,
            'edge_density': 0.003,
            'avg_degree': 36.0,
            'proxy_ratio': 0.06,
            'max_xs_corr': 0.12,
        },
        'profile_sparse_highimbalance': {
            'n_nodes_log10': 4.50,
            'pos_rate': 0.30,
            'sens_imbalance': 0.45,
            'ys_corr': 0.06,
            'edge_density': 0.002,
            'avg_degree': 60.0,
            'proxy_ratio': 0.22,
            'max_xs_corr': 0.45,
        },
        'profile_nba_hidden16': {
            'n_nodes_log10': 2.61,
            'pos_rate': 0.17,
            'sens_imbalance': 0.47,
            'ys_corr': 0.02,
            'edge_density': 0.20,
            'avg_degree': 82.0,
            'proxy_ratio': 0.20,
            'max_xs_corr': 0.40,
        },
    }

    scales = {
        'n_nodes_log10': 1.5,
        'pos_rate': 0.5,
        'sens_imbalance': 1.0,
        'ys_corr': 0.3,
        'edge_density': 0.05,
        'avg_degree': 80.0,
        'proxy_ratio': 1.0,
        'max_xs_corr': 1.0,
    }

    weights = {
        'n_nodes_log10': 0.8,
        'pos_rate': 1.0,
        'sens_imbalance': 1.2,
        'ys_corr': 2.0,
        'edge_density': 1.4,
        'avg_degree': 0.8,
        'proxy_ratio': 2.0,
        'max_xs_corr': 1.8,
    }

    distances = {
        name: _weighted_distance(sig, proto, weights, scales)
        for name, proto in prototypes.items()
    }
    sorted_pairs = sorted(distances.items(), key=lambda kv: kv[1])
    profile_name = sorted_pairs[0][0]
    best_dist = sorted_pairs[0][1]
    second_dist = sorted_pairs[1][1]
    margin = second_dist - best_dist

    # 安全门：high-imbalance profile 需要明显敏感分布偏斜，否则优先 low-proxy profile
    if (
        profile_name == 'profile_sparse_highimbalance'
        and sig.get('sens_imbalance', 0.0) < 0.18
    ):
        profile_name = 'profile_sparse_lowproxy'

    # 低置信度保护：相似度很接近时，优先选择中庸的 low-proxy profile
    if margin < 0.18:
        profile_name = 'profile_sparse_lowproxy'

    # NBA uses a dedicated temporary profile from the hidden=16 sweep.
    if sig.get('dataset') == 'nba':
        profile_name = 'profile_nba_hidden16'

    profile_params = {
        'profile_dense_balanced': {
            'lr': 0.007114466864080137,
            'dropout': 0.1,
            'alpha': 0.13,
            'beta': 0.68,
            'min_fair': 0.26,
            'warmup_steps': 300,
            'gate_fair_weight': 2.4,
            'res_val_f1_weight': 0.50,
            'res_val_roc_weight': 0.50,
            'constraint_mode': 'soft',
            'constraint_penalty_weight': 5.0,
            'min_auc': None,
            'min_f1': None,
            'max_parity': None,
            'max_equality': None,
            'use_group_fair_loss': False,
            'lambda_dp': 0.0,
            'lambda_eo': 0.0,
        },
        'profile_sparse_lowproxy': {
            'lr': 0.0023585982559417577,
            'weight_decay': 1e-4,
            'dropout': 0.27,
            'alpha': 0.23,
            'beta': 0.42,
            'min_fair': 0.72,
            'warmup_steps': 180,
            'gate_fair_weight': 7.0,
            'gate_fair_warmup_steps': 180,
            'res_val_f1_weight': 0.70,
            'res_val_roc_weight': 0.30,
            'constraint_mode': 'soft',
            'constraint_penalty_weight': 10.0,
            'min_auc': 89.0,
            'min_f1': 79.0,
            'max_parity': 3.2,
            'max_equality': 3.0,
            'use_group_fair_loss': True,
            'lambda_dp': 2.3,
            'lambda_eo': 2.0,
            'group_fair_warmup_steps': 220,
            'ref_on': True,
            'ref_ep': 240,
            'ref_lr': 0.18,
            'ref_fm': 3.2,
            'ref_tol': 0.35,
        },
        'profile_sparse_highimbalance': {
            'hidden': 8,
            'lr': 0.0014029487346801104,
            'dropout': 0.57,
            'alpha': 1.4,
            'beta': 2.2,
            'min_fair': 0.37,
            'warmup_steps': 220,
            'res_val_f1_weight': 0.40,
            'res_val_roc_weight': 0.60,
            'constraint_mode': 'soft',
            'constraint_penalty_weight': 8.0,
            'min_auc': 72.0,
            'min_f1': 87.4,
            'max_parity': 1.0,
            'max_equality': 1.0,
            'weight_decay': 0.003139594760822072,
            'lr_w': 3e-4,
            'stab_on': True,
            'stab_ep': 200,
            'stab_w': 2.8,
            'gate_tar': 0.81,
            'gate_mw': 5.0,
            'gate_vw': 1.3,
            'gate_hi': 0.89,
            'gate_hw': 6.0,
            'gate_ha': 0.76,
        },
        # EO-stability candidate (screen_trial=21), kept for reference:
        # result: AUC 70.26 / F1 70.49 / DP 4.88 / EO 2.00
        # 'profile_nba_hidden16_eo_stable': {
        #     'hidden': 16,
        #     'lr': 0.000938660354167432,
        #     'weight_decay': 4.52921668406068e-05,
        #     'dropout': 0.32,
        #     'alpha': 0.15000000000000002,
        #     'beta': 0.27999999999999997,
        #     'min_fair': 0.45,
        #     'warmup_steps': 120,
        #     'gate_fair_weight': 5.8,
        #     'gate_fair_warmup_steps': 120,
        #     'res_val_f1_weight': 0.75,
        #     'res_val_roc_weight': 0.25,
        #     'constraint_mode': 'soft',
        #     'constraint_penalty_weight': 35.0,
        #     'min_auc': 79.0,
        #     'min_f1': 76.0,
        #     'max_parity': 5.0,
        #     'max_equality': 3.0,
        #     'use_group_fair_loss': True,
        #     'lambda_dp': 5.4,
        #     'lambda_eo': 7.0,
        #     'group_fair_warmup_steps': 160,
        #     'sel_two': True,
        #     'sel_tol': 0.05,
        #     'ref_on': True,
        #     'ref_ep': 320,
        #     'ref_lr': 0.14,
        #     'ref_fm': 3.8,
        #     'ref_tol': 0.09,
        # },
        'profile_nba_hidden16': {
            'hidden': 16,
            'lr': 0.0006885460666066806,
            'weight_decay': 6.719379832019399e-06,
            'dropout': 0.26,
            'alpha': 0.12,
            'beta': 0.31,
            'min_fair': 0.47000000000000003,
            'warmup_steps': 120,
            'gate_fair_weight': 5.800000000000001,
            'gate_fair_warmup_steps': 140,
            'res_val_f1_weight': 0.85,
            'res_val_roc_weight': 0.15000000000000002,
            'constraint_mode': 'soft',
            'constraint_penalty_weight': 31.0,
            'min_auc': 79.0,
            'min_f1': 76.0,
            'max_parity': 5.0,
            'max_equality': 3.0,
            'use_group_fair_loss': True,
            'lambda_dp': 6.300000000000001,
            'lambda_eo': 3.5,
            'group_fair_warmup_steps': 100,
            'sel_two': True,
            'sel_tol': 0.07500000000000001,
            'ref_on': True,
            'ref_ep': 280,
            'ref_lr': 0.19,
            'ref_fm': 2.1,
            'ref_tol': 0.08499999999999999,
        },
    }

    if force_profile_name is not None:
        if force_profile_name not in profile_params:
            raise ValueError(f"Unknown forced profile: {force_profile_name}")
        profile_name = force_profile_name

    return {
        'name': profile_name,
        'params': profile_params[profile_name],
        'signature': sig,
        'routing': {
            'distances': distances,
            'best_dist': best_dist,
            'second_dist': second_dist,
            'margin': margin,
            'forced_profile_name': force_profile_name,
        }
    }


def apply_profile_to_args(args, profile, explicit_flags):
    applied = {}
    for k, v in profile['params'].items():
        if not explicit_flags.get(k, False):
            setattr(args, k, v)
            applied[k] = v

    # 保持 F1/ROC 权重一致
    if ('res_val_f1_weight' in applied) and (not explicit_flags.get('res_val_roc_weight', False)):
        args.res_val_roc_weight = 1.0 - float(args.res_val_f1_weight)
        applied['res_val_roc_weight'] = args.res_val_roc_weight

    return applied
