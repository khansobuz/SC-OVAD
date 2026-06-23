"""
Single file: SCNP + SCSEC + OVCD + PLOVAD MIL — UBnormal AUC
Run: python run_ubnormal.py
"""
import os, json, random
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
SAVE_DIR      = r"C:\Users\khanm\Desktop\lab_project\Plovad\ckpt_scnp"
LOG_PATH      = r"C:\Users\khanm\Desktop\lab_project\Plovad\log_scnp.log"

# ============================================================
#  CONFIG
# ============================================================
class Config:
    feat_dim   = 512
    max_seqlen = 450
    train_bs   = 64
    workers    = 0
    max_epoch  = 200
    lr         = 1e-4
    seed       = 2024
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    K          = 4      # SCNP prototypes
    hidden_dim = 256
    backbone   = "ViT-B/16"
    # loss weights — all positive contributions
    w_mil      = 1.0   # MIL binary loss (primary)
    w_norm     = 0.1   # SCNP reconstruction
    w_div      = 0.05  # SCNP diversity
    w_trans    = 0.05  # SCSEC transition
    w_ov       = 0.2   # OVCD open-vocab binary

cfg = Config()

# ============================================================
#  UTILS
# ============================================================
def setup_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_logger(path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    logger = logging.getLogger('SCNP')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(path, mode='w', encoding='utf-8'))
    sh = logging.StreamHandler()
    sh.stream = open(sh.stream.fileno(), mode='w', encoding='utf-8', closefd=False)
    logger.addHandler(sh)
    return logger

def process_feat(feat, length):
    T = len(feat)
    if T > length:
        idx = np.linspace(0, T-1, length, dtype=int)
        feat = feat[idx]
    elif T < length:
        pad = np.zeros((length - T, feat.shape[1]), dtype=np.float32)
        feat = np.concatenate([feat, pad], axis=0)
    return feat

# ============================================================
#  TEXT PROMPTING
# ============================================================
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
        if rel.startswith('val/'):
            rel = 'train/' + rel[4:]
        path = os.path.join(self.feat_prefix, rel.replace('/', os.sep))
        name = os.path.splitext(os.path.basename(path))[0]
        flag = os.path.basename(os.path.dirname(path))

        label   = 0.0 if flag == 'normal' else 1.0
        cls_name = 'normal' if flag == 'normal' else self.name2cls.get(name, 'normal')
        ano_idx  = self.cls_dict.get(cls_name, 0)

        feat = np.array(np.load(path), dtype=np.float32)
        if self.test_mode:
            return feat, label, ano_idx
        return process_feat(feat, self.max_seqlen), label, ano_idx

# ============================================================
#  MIL LOSS (PLOVAD style) — always positive (BCE)
# ============================================================
def mil_loss(logits, label, seq_len, device):
    bce  = nn.BCELoss()
    loss = torch.zeros(1, device=device)
    for i in range(logits.size(0)):
        sl  = seq_len[i].item()
        seg = logits[i, :sl]
        if label[i] > 0:
            k     = max(1, sl // 16)
            topk  = seg.topk(k).values
            loss  = loss + bce(topk, torch.ones_like(topk))
        else:
            loss  = loss + bce(seg, torch.zeros_like(seg))
    return loss / logits.size(0)

# ============================================================
#  SCNP
# ============================================================
class SCNP(nn.Module):
    def __init__(self, feat_dim, K):
        super().__init__()
        self.K = K
        self.omega_net  = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid())
        self.proto_heads = nn.ModuleList(
            [nn.Linear(feat_dim, feat_dim) for _ in range(K)])

    def forward(self, x):
        omega  = self.omega_net(x)                          # [B,T,1]
        s      = (omega * x).sum(1)                         # [B,D]
        protos = torch.stack([h(s) for h in self.proto_heads], dim=1)  # [B,K,D]
        diff   = x.unsqueeze(2) - protos.unsqueeze(1)      # [B,T,K,D]
        dist   = -diff.pow(2).sum(-1)                       # [B,T,K]
        alpha  = F.softmax(dist, dim=-1) * omega            # [B,T,K]
        alpha  = alpha / (alpha.sum(-1, keepdim=True) + 1e-8)
        x_hat  = (alpha.unsqueeze(-1) * protos.unsqueeze(1)).sum(2)  # [B,T,D]

        # losses — both positive
        L_norm = (omega.squeeze(-1) * (x - x_hat).pow(2).sum(-1)).mean()
        p_n    = F.normalize(protos, dim=-1)
        sim    = torch.bmm(p_n, p_n.transpose(1,2))         # [B,K,K]
        eye    = torch.eye(self.K, device=x.device).unsqueeze(0)
        L_div  = ((sim * (1 - eye)).abs()).mean()            # always >= 0

        return x_hat, s, omega.squeeze(-1), L_norm, L_div

# ============================================================
#  SCSEC
# ============================================================
class SCSEC(nn.Module):
    def __init__(self, feat_dim, M):
        super().__init__()
        self.proj    = nn.Linear(feat_dim, feat_dim)
        self.adapt   = nn.Linear(feat_dim*2, feat_dim)
        self.trans   = nn.Sequential(
            nn.Linear(feat_dim*2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim))

    def forward(self, residual, s, sem_protos, omega):
        v   = self.proj(residual)                           # [B,T,D]
        M   = sem_protos.size(0)
        B   = s.size(0)
        s_e = s.unsqueeze(1).expand(-1, M, -1)             # [B,M,D]
        p_e = sem_protos.unsqueeze(0).expand(B, -1, -1)    # [B,M,D]
        b_s = F.normalize(self.adapt(torch.cat([s_e, p_e], -1)), dim=-1)  # [B,M,D]

        gamma = F.softmax(torch.bmm(v, b_s.transpose(1,2)), dim=-1)  # [B,T,M]
        c_t   = torch.bmm(gamma, b_s)                      # [B,T,D]

        d_obs = c_t[:, 1:] - c_t[:, :-1]                  # [B,T-1,D]
        s_e2  = s.unsqueeze(1).expand(-1, c_t.size(1)-1, -1)
        d_exp = self.trans(torch.cat([c_t[:, :-1], s_e2], -1))

        e_t   = F.pad(torch.norm(d_exp - d_obs, dim=-1), (0,1))  # [B,T]
        a_sem = 1 - torch.bmm(
            F.normalize(c_t, dim=-1), b_s.transpose(1,2)).max(-1).values  # [B,T]
        a_sem = a_sem.clamp(0, 1)

        # transition loss — MSE always >= 0
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

    def forward(self, x_hat, c_t, sem_emb, seq_len, label, device, w_ov):
        z_s = F.relu(self.f_s(x_hat))
        z_c = F.relu(self.f_c(c_t))
        beta = torch.sigmoid(self.gate(torch.cat([z_s, z_c], -1)))
        z_t  = beta * z_s + (1 - beta) * z_c               # [B,T,H]
        logits = torch.sigmoid(self.out(z_t)).squeeze(-1)   # [B,T]

        # open-vocab: binary BCE on normal vs abnormal similarity — always positive
        B = z_t.size(0)
        vr = torch.stack([z_t[i, :seq_len[i]].mean(0) for i in range(B)])
        vr = F.normalize(vr, dim=-1)
        sem_h   = F.normalize(self.f_c(sem_emb), dim=-1)   # [M,H]
        sim     = vr @ sem_h.T                              # [B,M]
        abn_sim = sim[:, 1:].max(-1).values                 # [B]
        nor_sim = sim[:, 0]                                  # [B]
        score   = (abn_sim - nor_sim).clamp(-5, 5)          # [B] clamped for stability
        is_abn  = (label > 0).float().to(device)
        L_ov    = F.binary_cross_entropy_with_logits(score, is_abn)

        return logits, L_ov

# ============================================================
#  FULL MODEL
# ============================================================
class FullModel(nn.Module):
    def __init__(self, feat_dim, K, M, hidden_dim):
        super().__init__()
        self.scnp  = SCNP(feat_dim, K)
        self.scsec = SCSEC(feat_dim, M)
        self.ovcd  = OVCD(feat_dim, hidden_dim)

    def forward(self, x, seq_len, cls_list, label, device, cfg):
        # SCNP
        x_hat, s, omega, L_norm, L_div = self.scnp(x)

        # SCSEC
        residual = x - x_hat
        e_t, a_sem, c_t, b_s, L_trans = self.scsec(residual, s, cls_list, omega)

        # OVCD
        logits, L_ov = self.ovcd(x_hat, c_t, cls_list, seq_len, label, device, cfg.w_ov)

        # MIL
        L_mil = mil_loss(logits, label, seq_len, device)

        # total loss — all terms positive
        L_mil   = L_mil.clamp(max=10)
        L_norm  = L_norm.clamp(max=10)
        L_div   = L_div.clamp(max=10)
        L_trans = L_trans.clamp(max=10)
        L_ov    = L_ov.clamp(max=10)
        loss = (cfg.w_mil   * L_mil  +
                cfg.w_norm  * L_norm +
                cfg.w_div   * L_div  +
                cfg.w_trans * L_trans+
                cfg.w_ov    * L_ov)

        # anomaly score — weighted combination
        a_score = logits + 0.5 * e_t.clamp(0,1) + 0.3 * a_sem.clamp(0,1)

        return a_score, loss

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
def test_auc(loader, model, gt, cls_list, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for v_input, label, _ in loader:
            if isinstance(v_input, np.ndarray):
                v_input = torch.from_numpy(v_input).float().to(device)
            else:
                v_input = v_input.clone().detach().float().to(device)
            if v_input.dim() == 2:
                v_input = v_input.unsqueeze(0)
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
#  MAIN
# ============================================================
def main():
    setup_seed(cfg.seed)
    logger = get_logger(LOG_PATH)
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = cfg.device
    logger.info(f"Device: {device}")

    logger.info("Loading CLIP text embeddings...")
    cls_list, cls_dict = text_prompting(CLS_LIST_PATH, cfg.backbone, device)
    cls_list = cls_list.to(device)
    cfg.M = cls_list.size(0)
    logger.info(f"Classes ({cfg.M}): {list(cls_dict.keys())}")

    with open(NAME2CLS_PATH) as f:
        name2cls = json.load(f)

    train_data = UBDataset(FEAT_PREFIX, TRAIN_LIST, name2cls, cls_dict, cfg.max_seqlen, False)
    test_data  = UBDataset(FEAT_PREFIX, TEST_LIST,  name2cls, cls_dict, cfg.max_seqlen, True)
    train_loader = DataLoader(train_data, batch_size=cfg.train_bs, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_data,  batch_size=1,            shuffle=False, num_workers=0)

    gt = np.load(GT_PATH)
    logger.info(f"Train: {len(train_data)} | Test: {len(test_data)} | GT: {len(gt)}")

    model = FullModel(cfg.feat_dim, cfg.K, cfg.M, cfg.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epoch, eta_min=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {n_params/1e6:.2f}M")

    best_auc = 0.0
    for epoch in range(cfg.max_epoch):
        loss    = train_epoch(train_loader, model, optimizer, cls_list, device, cfg)
        roc_auc = test_auc(test_loader, model, gt, cls_list, device)
        scheduler.step()

        is_best = roc_auc >= best_auc
        if is_best:
            best_auc = roc_auc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pkl"))

        logger.info(f"[Epoch {epoch+1:3d}/{cfg.max_epoch}] Loss: {loss:.4f} | AUC: {roc_auc:.4f} | Best: {best_auc:.4f}" + (" *BEST*" if is_best else ""))

    logger.info(f"\n=== FINAL BEST AUC: {best_auc:.4f} ===")

if __name__ == '__main__':
    main()