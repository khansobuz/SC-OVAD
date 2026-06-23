"""
Ablation Study for UBnormal
Runs 4 experiments and prints comparison table at the end.
Run: python ablation_study.py
"""

import os, json, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import auc, roc_curve
import clip
import logging

# ============================================================
#  PATHS
# ============================================================
FEAT_PREFIX   = r"C:\Users\khanm\Desktop\lab_project\Plovad\features"
LIST_DIR      = r"C:\Users\khanm\Desktop\lab_project\Plovad\src\list\ubnormal"
TRAIN_LIST    = os.path.join(LIST_DIR, "ub-vit-train.list")
TEST_LIST     = os.path.join(LIST_DIR, "ub-vit-test.list")
GT_PATH       = os.path.join(LIST_DIR, "gt.npy")
NAME2CLS_PATH = os.path.join(LIST_DIR, "name2cls.json")
CLS_LIST_PATH = r"C:\Users\khanm\Desktop\lab_project\Plovad\src\list\prompt\class_ubnormal.txt"
SAVE_DIR      = r"C:\Users\khanm\Desktop\lab_project\Plovad\ckpt_ablation"
LOG_PATH      = r"C:\Users\khanm\Desktop\lab_project\Plovad\ablation_log.txt"

# ============================================================
#  CONFIG
# ============================================================
class Config:
    feat_dim   = 512
    max_seqlen = 450
    train_bs   = 64
    workers    = 0
    max_epoch  = 50
    lr         = 1e-4
    seed       = 2024
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    K          = 4
    hidden_dim = 256
    backbone   = "ViT-B/16"
    w_mil      = 1.0
    w_norm     = 0.1
    w_div      = 0.05
    w_trans    = 0.05
    w_ov       = 0.2
    # ablation flags
    use_scnp   = True
    use_scsec  = True
    use_ovcd   = True

# ============================================================
#  UTILS
# ============================================================
def setup_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def process_feat(feat, length):
    T = len(feat)
    if T > length:
        idx = np.linspace(0, T-1, length, dtype=int)
        feat = feat[idx]
    elif T < length:
        pad = np.zeros((length - T, feat.shape[1]), dtype=np.float32)
        feat = np.concatenate([feat, pad], axis=0)
    return feat

def text_prompting(cls_path, backbone, device):
    with open(cls_path) as f:
        cls_names = [l.strip() for l in f if l.strip()]
    model, _ = clip.load(backbone, device=device)
    model.eval()
    cls_dict = {}
    feats = []
    with torch.no_grad():
        for i, name in enumerate(cls_names):
            tok  = clip.tokenize([f"a video of {name}"]).to(device)
            feat = model.encode_text(tok).float()
            feat = F.normalize(feat, dim=-1)
            feats.append(feat)
            cls_dict[name] = i
    cls_list = torch.cat(feats, dim=0)
    del model; torch.cuda.empty_cache()
    return cls_list, cls_dict

# ============================================================
#  DATASET
# ============================================================
class UBDataset(Dataset):
    def __init__(self, feat_prefix, list_file, name2cls, cls_dict, max_seqlen, test_mode=False):
        self.feat_prefix = feat_prefix
        self.max_seqlen  = max_seqlen
        self.test_mode   = test_mode
        self.name2cls    = name2cls
        self.cls_dict    = cls_dict
        self.data_list   = [l.strip() for l in open(list_file) if l.strip()]

    def __len__(self): return len(self.data_list)

    def __getitem__(self, index):
        rel = self.data_list[index]
        if rel.startswith('val/'): rel = 'train/' + rel[4:]
        path = os.path.join(self.feat_prefix, rel.replace('/', os.sep))
        name = os.path.splitext(os.path.basename(path))[0]
        flag = os.path.basename(os.path.dirname(path))
        label    = 0.0 if flag == 'normal' else 1.0
        cls_name = 'normal' if flag == 'normal' else self.name2cls.get(name, 'normal')
        ano_idx  = self.cls_dict.get(cls_name, 0)
        feat = np.array(np.load(path), dtype=np.float32)
        if self.test_mode:
            return feat, label, ano_idx
        return process_feat(feat, self.max_seqlen), label, ano_idx

# ============================================================
#  MIL LOSS
# ============================================================
def mil_loss(logits, label, seq_len, device):
    bce  = nn.BCELoss()
    loss = torch.zeros(1, device=device)
    for i in range(logits.size(0)):
        sl  = seq_len[i].item()
        seg = logits[i, :sl]
        if label[i] > 0:
            k    = max(1, sl // 16)
            topk = seg.topk(k).values
            loss = loss + bce(topk, torch.ones_like(topk))
        else:
            loss = loss + bce(seg, torch.zeros_like(seg))
    return loss / logits.size(0)

# ============================================================
#  SCNP
# ============================================================
class SCNP(nn.Module):
    def __init__(self, feat_dim, K):
        super().__init__()
        self.K = K
        self.omega_net   = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid())
        self.proto_heads = nn.ModuleList(
            [nn.Linear(feat_dim, feat_dim) for _ in range(K)])

    def forward(self, x):
        omega  = self.omega_net(x)
        s      = (omega * x).sum(1)
        protos = torch.stack([h(s) for h in self.proto_heads], dim=1)
        diff   = x.unsqueeze(2) - protos.unsqueeze(1)
        dist   = -diff.pow(2).sum(-1)
        alpha  = F.softmax(dist, dim=-1) * omega
        alpha  = alpha / (alpha.sum(-1, keepdim=True) + 1e-8)
        x_hat  = (alpha.unsqueeze(-1) * protos.unsqueeze(1)).sum(2)
        L_norm = (omega.squeeze(-1) * (x - x_hat).pow(2).sum(-1)).mean()
        p_n    = F.normalize(protos, dim=-1)
        sim    = torch.bmm(p_n, p_n.transpose(1,2))
        eye    = torch.eye(self.K, device=x.device).unsqueeze(0)
        L_div  = ((sim * (1 - eye)).abs()).mean()
        return x_hat, s, omega.squeeze(-1), L_norm, L_div

# ============================================================
#  SCSEC
# ============================================================
class SCSEC(nn.Module):
    def __init__(self, feat_dim, M):
        super().__init__()
        self.proj  = nn.Linear(feat_dim, feat_dim)
        self.adapt = nn.Linear(feat_dim*2, feat_dim)
        self.trans = nn.Sequential(
            nn.Linear(feat_dim*2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim))

    def forward(self, residual, s, sem_protos, omega):
        v   = self.proj(residual)
        M   = sem_protos.size(0); B = s.size(0)
        s_e = s.unsqueeze(1).expand(-1, M, -1)
        p_e = sem_protos.unsqueeze(0).expand(B, -1, -1)
        b_s = F.normalize(self.adapt(torch.cat([s_e, p_e], -1)), dim=-1)
        gamma = F.softmax(torch.bmm(v, b_s.transpose(1,2)), dim=-1)
        c_t   = torch.bmm(gamma, b_s)
        d_obs = c_t[:, 1:] - c_t[:, :-1]
        s_e2  = s.unsqueeze(1).expand(-1, c_t.size(1)-1, -1)
        d_exp = self.trans(torch.cat([c_t[:, :-1], s_e2], -1))
        e_t   = F.pad(torch.norm(d_exp - d_obs, dim=-1), (0,1))
        a_sem = (1 - torch.bmm(
            F.normalize(c_t,dim=-1), b_s.transpose(1,2)).max(-1).values).clamp(0,1)
        L_trans = (omega[:, :-1] * (d_exp - d_obs).pow(2).sum(-1)).mean()
        return e_t, a_sem, c_t, b_s, L_trans

# ============================================================
#  OVCD
# ============================================================
class OVCD(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()
        self.f_s  = nn.Linear(feat_dim, hidden_dim)
        self.f_c  = nn.Linear(feat_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim*2, hidden_dim)
        self.out  = nn.Linear(hidden_dim, 1)

    def forward(self, x_hat, c_t, sem_emb, seq_len, label, device):
        z_s  = F.relu(self.f_s(x_hat))
        z_c  = F.relu(self.f_c(c_t))
        beta = torch.sigmoid(self.gate(torch.cat([z_s, z_c], -1)))
        z_t  = beta * z_s + (1 - beta) * z_c
        logits = torch.sigmoid(self.out(z_t)).squeeze(-1)
        B  = z_t.size(0)
        vr = torch.stack([z_t[i, :seq_len[i]].mean(0) for i in range(B)])
        vr = F.normalize(vr, dim=-1)
        sem_h   = F.normalize(self.f_c(sem_emb), dim=-1)
        sim     = vr @ sem_h.T
        score   = (sim[:, 1:].max(-1).values - sim[:, 0]).clamp(-5, 5)
        is_abn  = (label > 0).float().to(device)
        L_ov    = F.binary_cross_entropy_with_logits(score, is_abn)
        return logits, z_t, L_ov

# ============================================================
#  FULL MODEL with ablation flags
# ============================================================
class FullModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        M = cfg.M
        self.use_scnp  = cfg.use_scnp
        self.use_scsec = cfg.use_scsec
        self.use_ovcd  = cfg.use_ovcd
        self.feat_dim  = cfg.feat_dim

        if self.use_scnp:
            self.scnp = SCNP(cfg.feat_dim, cfg.K)
        if self.use_scsec:
            self.scsec = SCSEC(cfg.feat_dim, M)
        if self.use_ovcd:
            self.ovcd = OVCD(cfg.feat_dim, cfg.hidden_dim)
        else:
            # simple MLP classifier when OVCD is off
            self.classifier = nn.Sequential(
                nn.Linear(cfg.feat_dim, cfg.hidden_dim),
                nn.ReLU(), nn.Linear(cfg.hidden_dim, 1), nn.Sigmoid())

    def forward(self, x, seq_len, cls_list, label, device, cfg):
        B, T, D = x.shape
        losses = []

        # --- SCNP or passthrough ---
        if self.use_scnp:
            x_hat, s, omega, L_norm, L_div = self.scnp(x)
            losses.append(cfg.w_norm * L_norm.clamp(max=10))
            losses.append(cfg.w_div  * L_div.clamp(max=10))
            residual = x - x_hat
        else:
            x_hat = x
            s     = x.mean(dim=1)             # simple mean as scene repr
            omega = torch.ones(B, T, device=device)
            residual = x

        # --- SCSEC or zeros ---
        if self.use_scsec:
            e_t, a_sem, c_t, b_s, L_trans = self.scsec(residual, s, cls_list, omega)
            losses.append(cfg.w_trans * L_trans.clamp(max=10))
        else:
            e_t   = torch.zeros(B, T, device=device)
            a_sem = torch.zeros(B, T, device=device)
            c_t   = residual

        # --- OVCD or simple classifier ---
        if self.use_ovcd:
            logits, z_t, L_ov = self.ovcd(x_hat, c_t, cls_list, seq_len, label, device)
            losses.append(cfg.w_ov * L_ov.clamp(max=10))
        else:
            logits = self.classifier(x_hat).squeeze(-1)  # [B,T]

        # MIL loss
        L_mil = mil_loss(logits, label, seq_len, device)
        losses.append(cfg.w_mil * L_mil.clamp(max=10))

        total_loss = sum(losses)

        # anomaly score
        a_score = logits + 0.5 * e_t.clamp(0,1) + 0.3 * a_sem

        return a_score, total_loss

# ============================================================
#  TRAIN
# ============================================================
def train_epoch(loader, model, optimizer, cls_list, device, cfg):
    model.train()
    total = 0
    for v_input, label, _ in loader:
        v_input = torch.from_numpy(np.array(v_input)).float().to(device)
        label   = torch.from_numpy(np.array(label)).float().to(device)
        seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, dim=1)
        v_input = v_input[:, :torch.max(seq_len), :]
        _, loss = model(v_input, seq_len, cls_list, label, device, cfg)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)

# ============================================================
#  TEST
# ============================================================
def test_auc(loader, model, gt, cls_list, device, cfg):
    model.eval()
    preds = []
    with torch.no_grad():
        for v_input, label, _ in loader:
            if isinstance(v_input, np.ndarray):
                v_input = torch.from_numpy(v_input).float().to(device)
            else:
                v_input = v_input.clone().detach().float().to(device)
            if v_input.dim() == 2: v_input = v_input.unsqueeze(0)
            T = v_input.size(1)
            seq_len = torch.tensor([T], device=device)
            label_t = torch.zeros(1, device=device)
            scores, _ = model(v_input, seq_len, cls_list, label_t, device, cfg)
            frame_scores = np.repeat(scores[0, :T].cpu().numpy(), 16)
            preds.extend(frame_scores.tolist())
    preds  = np.array(preds)
    gt_arr = np.array(list(gt))
    n      = min(len(preds), len(gt_arr))
    fpr, tpr, _ = roc_curve(gt_arr[:n], preds[:n])
    return auc(fpr, tpr)

# ============================================================
#  RUN ONE EXPERIMENT
# ============================================================
def run_experiment(exp_name, cfg, cls_list, train_loader, test_loader, gt, logger):
    device = cfg.device
    setup_seed(cfg.seed)

    logger.info(f"\n{'='*60}")
    logger.info(f"  EXPERIMENT: {exp_name}")
    logger.info(f"  use_scnp={cfg.use_scnp} | use_scsec={cfg.use_scsec} | use_ovcd={cfg.use_ovcd}")
    logger.info(f"{'='*60}")

    model     = FullModel(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epoch, eta_min=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Params: {n_params/1e6:.2f}M")

    best_auc = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(cfg.max_epoch):
        loss    = train_epoch(train_loader, model, optimizer, cls_list, device, cfg)
        roc_auc = test_auc(test_loader, model, gt, cls_list, device, cfg)
        scheduler.step()

        is_best = roc_auc >= best_auc
        if is_best:
            best_auc = roc_auc
            safe_name = exp_name.replace(" ", "_").replace("/", "").replace("+", "plus")
            save_path = os.path.join(SAVE_DIR, f"{safe_name}_best.pkl")
            torch.save(model.state_dict(), save_path)

        logger.info(f"  [{exp_name}] Epoch {epoch+1:3d}/{cfg.max_epoch} | "
                    f"Loss: {loss:.4f} | AUC: {roc_auc:.4f} | Best: {best_auc:.4f}"
                    + (" *BEST*" if is_best else ""))

    logger.info(f"\n  [{exp_name}] FINAL BEST AUC: {best_auc:.4f}")
    return best_auc

# ============================================================
#  MAIN — run all ablations
# ============================================================
def main():
    # setup logger
    os.makedirs(os.path.dirname(LOG_PATH) if os.path.dirname(LOG_PATH) else '.', exist_ok=True)
    logger = logging.getLogger('Ablation')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(LOG_PATH, mode='w', encoding='utf-8'))
    sh = logging.StreamHandler()
    logger.addHandler(sh)

    cfg    = Config()
    device = cfg.device
    logger.info(f"Device: {device}")

    # load CLIP once
    logger.info("Loading CLIP text embeddings...")
    cls_list, cls_dict = text_prompting(CLS_LIST_PATH, cfg.backbone, device)
    cls_list = cls_list.to(device)
    cfg.M    = cls_list.size(0)
    logger.info(f"Classes: {cfg.M}")

    with open(NAME2CLS_PATH) as f:
        name2cls = json.load(f)

    train_data   = UBDataset(FEAT_PREFIX, TRAIN_LIST, name2cls, cls_dict, cfg.max_seqlen, False)
    test_data    = UBDataset(FEAT_PREFIX, TEST_LIST,  name2cls, cls_dict, cfg.max_seqlen, True)
    train_loader = DataLoader(train_data, batch_size=cfg.train_bs, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_data,  batch_size=1,            shuffle=False, num_workers=0)
    gt = np.load(GT_PATH)

    # define 4 ablation experiments
    experiments = [
        ("Full Model (SCNP+SCSEC+OVCD)", True,  True,  True),
        ("w/o SCNP",                      False, True,  True),
        ("w/o SCSEC",                     True,  False, True),
        ("w/o OVCD",                      True,  True,  False),
    ]

    results = {}
    for exp_name, use_scnp, use_scsec, use_ovcd in experiments:
        exp_cfg           = copy.copy(cfg)
        exp_cfg.use_scnp  = use_scnp
        exp_cfg.use_scsec = use_scsec
        exp_cfg.use_ovcd  = use_ovcd

        best = run_experiment(
            exp_name, exp_cfg, cls_list,
            train_loader, test_loader, gt, logger)
        results[exp_name] = best

    # print final comparison table
    logger.info("\n" + "="*60)
    logger.info("  ABLATION STUDY RESULTS")
    logger.info("="*60)
    logger.info(f"  {'Model':<35} | {'AUC':>8} | {'vs Full':>8}")
    logger.info(f"  {'-'*35}-+-{'-'*8}-+-{'-'*8}")
    full_auc = results["Full Model (SCNP+SCSEC+OVCD)"]
    for name, auc_val in results.items():
        diff = auc_val - full_auc
        sign = "+" if diff >= 0 else ""
        logger.info(f"  {name:<35} | {auc_val:>8.4f} | {sign}{diff:>7.4f}")
    logger.info("="*60)
    logger.info(f"  Results saved to: {LOG_PATH}")

if __name__ == '__main__':
    main()