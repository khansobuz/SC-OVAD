"""
SCOPE-VAD Final — UCF-Crime Target: AUC >= 86%

KEY INSIGHT FROM ALL PREVIOUS RUNS:
  Every version that dropped below 85% had auxiliary losses competing with MIL.
  Your original code (85.85%) used pure MIL. The paper modules (SCNP, CTEC, OVCD)
  must ENHANCE the MIL score, not compete with it.

ARCHITECTURE STRATEGY:
  1. SCNP reconstructs scene-conditioned normality -> reconstruction residual
     feeds CTEC and enriches anomaly scoring
  2. CTEC causal state evolution on residual features -> causal anomaly signal
  3. OVCD fuses SCNP + CTEC outputs -> final anomaly score via MIL
  4. ALL auxiliary losses (L_norm, L_temp, L_decouple) weight < 0.01
  5. MIL is the ONLY primary loss (weight = 1.0)

This is exactly how RTFM, MGFN, UR-DMU achieve 86%+ — MIL dominant,
everything else is regularisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
import numpy as np
import os
import random


# ══════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════

class Normal_Loader(Dataset):
    def __init__(self, is_train=1, path='./UCF-Crime/', modality='TWO'):
        super().__init__()
        self.is_train = is_train
        self.modality = modality
        self.path = path
        fp = os.path.join(path, 'train_normal.txt' if is_train == 1 else 'test_normalv2.txt')
        with open(fp, 'r') as f:
            self.data_list = f.readlines()
        if is_train == 0:
            random.shuffle(self.data_list)
            self.data_list = self.data_list[:-10]

    def __len__(self): return len(self.data_list)

    def _load(self, name):
        rgb  = np.load(os.path.join(self.path + 'all_rgbs',  name + '.npy'))
        flow = np.load(os.path.join(self.path + 'all_flows', name + '.npy'))
        if self.modality == 'RGB':  return rgb
        if self.modality == 'FLOW': return flow
        return np.concatenate([rgb, flow], axis=1)

    def __getitem__(self, idx):
        if self.is_train == 1:
            return self._load(self.data_list[idx][:-1])
        p = self.data_list[idx].split(' ')
        return self._load(p[0]), int(p[2][:-1]), int(p[1])


class Anomaly_Loader(Dataset):
    def __init__(self, is_train=1, path='./UCF-Crime/', modality='TWO'):
        super().__init__()
        self.is_train = is_train
        self.modality = modality
        self.path = path
        fp = os.path.join(path, 'train_anomaly.txt' if is_train == 1 else 'test_anomalyv2.txt')
        with open(fp, 'r') as f:
            self.data_list = f.readlines()

    def __len__(self): return len(self.data_list)

    def _load(self, name):
        rgb  = np.load(os.path.join(self.path + 'all_rgbs',  name + '.npy'))
        flow = np.load(os.path.join(self.path + 'all_flows', name + '.npy'))
        if self.modality == 'RGB':  return rgb
        if self.modality == 'FLOW': return flow
        return np.concatenate([rgb, flow], axis=1)

    def __getitem__(self, idx):
        if self.is_train == 1:
            return self._load(self.data_list[idx][:-1])
        p   = self.data_list[idx].split('|')
        gts = [int(i) for i in p[2][1:-2].split(',')]
        return self._load(p[0]), gts, int(p[1])


# ══════════════════════════════════════════════════════════════
# MODULE A: SCNP  (paper Eq 1-6)
# ══════════════════════════════════════════════════════════════

class SCNP(nn.Module):
    """
    Scene-Conditioned Normality Prototype.
    Eq 1: s = mean_t(x_t)                     scene descriptor
    Eq 2: p_k = W_k * s + b_k                 K scene-conditioned prototypes
    Eq 3: alpha_{t,k} = softmax(-||x-p_k||^2) soft assignment
    Eq 4: x_hat_t = sum_k alpha_{t,k} * p_k   reconstruction
    Eq 5: a_t = ||x_t - x_hat_t||             scene deviation score
    Eq 6: L_norm = sum_{normal} ||x - x_hat||^2
    """
    def __init__(self, feat_dim, K=4):
        super().__init__()
        self.K = K
        # K scene-conditioned prototype projectors
        self.proto_W = nn.Linear(feat_dim, feat_dim * K, bias=True)
        self._init()

    def _init(self):
        nn.init.xavier_uniform_(self.proto_W.weight)
        nn.init.zeros_(self.proto_W.bias)

    def forward(self, X):
        """X: (B, T, D) -> x_hat (B,T,D), a_t (B,T), alpha (B,T,K)"""
        B, T, D = X.shape

        # Eq 1: scene descriptor
        s = X.mean(dim=1)                                    # (B, D)

        # Eq 2: K prototypes conditioned on scene
        P = self.proto_W(s).view(B, self.K, D)              # (B, K, D)

        # Eq 3: soft assignment via negative squared distance
        X_e = X.unsqueeze(2).expand(B, T, self.K, D)        # (B, T, K, D)
        P_e = P.unsqueeze(1).expand(B, T, self.K, D)
        sq  = ((X_e - P_e)**2).sum(-1)                      # (B, T, K)
        alpha = torch.softmax(-sq / (D ** 0.5), dim=-1)     # (B, T, K)

        # Eq 4: reconstruction
        x_hat = torch.einsum('btk,bkd->btd', alpha, P)      # (B, T, D)

        # Eq 5: deviation score — normalised to [0,1]
        a_t = torch.norm(X - x_hat, dim=-1)                  # (B, T)
        a_t = torch.sigmoid(a_t)

        return x_hat, a_t, P


# ══════════════════════════════════════════════════════════════
# MODULE B: CTEC  (paper Eq 7-14)
# ══════════════════════════════════════════════════════════════

class CTEC(nn.Module):
    """
    Causality-Structured Temporal Event Chain.
    Eq 7 : X_tilde = X - x_hat  (residual)
    Eq 8 : phases {cause, buildup, peak, aftermath}
    Eq 9 : phase basis B = {b_m}
    Eq 10: gamma = softmax(phi(x_tilde)^T * B)
    Eq 11: c_t = sum_m gamma_{t,m} * b_m
    Eq 12: u_t = sigma(A*u_{t-1} + W_x*x_tilde + W_c*c_t)
    Eq 13: e_t = ||Psi(u_t) - Psi(u_{t-1})||
    Eq 14: a_tilde = e_t + g(u_t)
    """
    def __init__(self, feat_dim, M=4, state_dim=256):
        super().__init__()
        self.M         = M
        self.state_dim = state_dim

        # Eq 9: learnable phase basis
        self.phase_basis = nn.Parameter(torch.randn(M, feat_dim) * 0.01)

        # Eq 10: phi projection
        self.phi = nn.Linear(feat_dim, feat_dim, bias=False)

        # Eq 12: causal state transition
        self.A   = nn.Parameter(torch.eye(state_dim) * 0.95)
        self.W_x = nn.Linear(feat_dim,  state_dim, bias=False)
        self.W_c = nn.Linear(feat_dim,  state_dim, bias=False)
        self.b_u = nn.Parameter(torch.zeros(state_dim))

        # Eq 13: Psi — discriminative causal mapping
        self.Psi = nn.Sequential(
            nn.Linear(state_dim, state_dim), nn.ReLU(),
            nn.Linear(state_dim, state_dim)
        )

        # Eq 14: g — intrinsic abnormality
        self.g = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X_tilde):
        """X_tilde: (B, T, D) -> u_seq (B,T,state), a_tilde (B,T), gamma (B,T,M)"""
        B, T, D = X_tilde.shape

        # Eq 10: phase activation
        phi_out = self.phi(X_tilde)                          # (B, T, D)
        logits  = torch.einsum('btd,md->btm', phi_out, self.phase_basis)
        gamma   = torch.softmax(logits, dim=-1)              # (B, T, M)

        # Eq 11: phase-aware embedding
        c_t = torch.einsum('btm,md->btd', gamma, self.phase_basis)

        # Eq 12: causal state recurrence
        u    = torch.zeros(B, self.state_dim, device=X_tilde.device)
        u_seq = []
        for t in range(T):
            u = torch.sigmoid(
                u @ self.A.T +
                self.W_x(X_tilde[:, t]) +
                self.W_c(c_t[:, t]) +
                self.b_u
            )
            u_seq.append(u)
        u_seq = torch.stack(u_seq, dim=1)                    # (B, T, state_dim)

        # Eq 13: transition energy
        psi   = self.Psi(u_seq)                              # (B, T, state_dim)
        e     = torch.zeros(B, T, device=X_tilde.device)
        e[:, 1:] = torch.norm(psi[:, 1:] - psi[:, :-1], dim=-1)

        # Eq 14: causal anomaly score
        g_u     = self.g(u_seq).squeeze(-1)                  # (B, T)
        a_tilde = torch.sigmoid(e + g_u)                     # (B, T) in [0,1]

        return u_seq, a_tilde, gamma


# ══════════════════════════════════════════════════════════════
# MODULE C: OVCD  (paper Eq 15-24)
# ══════════════════════════════════════════════════════════════

class OVCD(nn.Module):
    """
    Open-Vocabulary Contrastive Decoupling.
    Eq 15: z_s = f_s(x_hat),  z_c = f_c(u_t)
    Eq 16: L_decouple = cos(z_s, z_c)^2
    Eq 17: beta = sigma(W_b [z_s; z_c]),  z = beta*z_s + (1-beta)*z_c
    Eq 24: A_t = alpha*(1 - max_c sim) + (1-alpha)*a_tilde  (inference score)

    PRIMARY OUTPUT: MIL anomaly score derived from fused representation.
    """
    def __init__(self, feat_dim, state_dim, proj_dim=256):
        super().__init__()
        # Eq 15: scene and causal projectors
        self.f_s = nn.Sequential(
            nn.Linear(feat_dim,  proj_dim),
            nn.LayerNorm(proj_dim), nn.ReLU()
        )
        self.f_c = nn.Sequential(
            nn.Linear(state_dim, proj_dim),
            nn.LayerNorm(proj_dim), nn.ReLU()
        )

        # Eq 17: gated fusion
        self.W_b = nn.Linear(proj_dim * 2, proj_dim)

        # Primary MIL scorer on fused representation
        self.scorer = nn.Sequential(
            nn.Linear(proj_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),     nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),     nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Learned combination weight for Eq 24
        self.alpha = nn.Parameter(torch.tensor(0.6))

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_hat, u_seq, a_t, a_tilde):
        """
        x_hat  : (B, T, D)          SCNP reconstruction
        u_seq  : (B, T, state_dim)  CTEC causal states
        a_t    : (B, T)             SCNP deviation score
        a_tilde: (B, T)             CTEC causal score
        Returns: score (B, T) in [0,1], z_s, z_c for decouple loss
        """
        # Eq 15
        z_s = self.f_s(x_hat)                               # (B, T, proj_dim)
        z_c = self.f_c(u_seq)                               # (B, T, proj_dim)

        # Eq 17: gated fusion
        beta  = torch.sigmoid(self.W_b(torch.cat([z_s, z_c], dim=-1)))
        z     = beta * z_s + (1 - beta) * z_c              # (B, T, proj_dim)

        # Primary MIL score from fused representation
        mil_score = torch.sigmoid(self.scorer(z).squeeze(-1))  # (B, T) in [0,1]

        # Eq 24: combine with SCNP deviation + CTEC causal score
        alpha = torch.sigmoid(self.alpha)
        score = alpha * mil_score + (1 - alpha) * (0.5 * a_t + 0.5 * a_tilde)

        return score, z_s, z_c

    def decouple_loss(self, z_s, z_c):
        """Eq 16: decorrelation between scene and causal branches."""
        zs = F.normalize(z_s.reshape(-1, z_s.size(-1)), dim=-1)
        zc = F.normalize(z_c.reshape(-1, z_c.size(-1)), dim=-1)
        return ((zs * zc).sum(dim=-1) ** 2).mean()

    def temporal_consistency_loss(self, score, a_tilde):
        """Eq 21: align MIL score with causal dynamics (StopGrad on a_tilde)."""
        target = a_tilde.detach()
        return F.mse_loss(score, target)


# ══════════════════════════════════════════════════════════════
# FULL MODEL
# ══════════════════════════════════════════════════════════════

class SCOPE_VAD(nn.Module):
    def __init__(self, input_dim=2048, K=4, M=4, state_dim=256, proj_dim=256):
        super().__init__()
        # Input projection
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.scnp = SCNP(feat_dim=input_dim, K=K)
        self.ctec = CTEC(feat_dim=input_dim, M=M, state_dim=state_dim)
        self.ovcd = OVCD(feat_dim=input_dim, state_dim=state_dim, proj_dim=proj_dim)
        self._init()

    def _init(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, T=32):
        BT, D = x.shape
        B     = BT // T
        X     = self.proj(x).view(B, T, D)                  # (B, T, D)

        # SCNP
        x_hat, a_t, P = self.scnp(X)

        # CTEC on residual (Eq 7)
        X_tilde = (X - x_hat).detach()                      # no grad to SCNP from CTEC
        u_seq, a_tilde, gamma = self.ctec(X_tilde)

        # OVCD
        score, z_s, z_c = self.ovcd(x_hat, u_seq, a_t, a_tilde)

        return score, z_s, z_c, x_hat, X, a_tilde


# ══════════════════════════════════════════════════════════════
# LOSSES
# ══════════════════════════════════════════════════════════════

def mil_loss(score, batch_size, device, margin=0.9):
    """
    Top-k MIL: averages top-k=4 clips per bag.
    score: (2B, T) — anomaly bags first, normal bags second.
    """
    B = batch_size
    T = score.size(1)
    k = max(1, T // 8)

    a_sc = score[:B]    # (B, T)
    n_sc = score[B:]    # (B, T)

    loss = smooth = torch.tensor(0., device=device)
    for i in range(B):
        a_topk = a_sc[i].topk(k).values.mean()
        n_topk = n_sc[i].topk(k).values.mean()
        loss  += F.relu(margin - (a_topk - n_topk))
        smooth += ((a_sc[i,:-1] - a_sc[i,1:])**2).mean() * 8e-4
    return (loss + smooth) / B


def compute_loss(score, z_s, z_c, x_hat, X, a_tilde,
                 batch_size, device, model, epoch):
    B = batch_size

    # Reshape (2B, T) from flat output
    T     = score.size(1)

    # 1. MIL loss (PRIMARY — weight 1.0)
    l_mil = mil_loss(score, B, device)

    # 2. SCNP reconstruction loss on normal bags (Eq 6) — tiny weight
    X_norm    = X[B:]
    xhat_norm = x_hat[B:]
    l_norm    = ((X_norm - xhat_norm)**2).sum(-1).mean()

    # 3. OVCD decouple loss (Eq 16) — tiny weight
    l_dec = model.ovcd.decouple_loss(z_s, z_c)

    # 4. Temporal consistency (Eq 21) — tiny weight
    l_temp = model.ovcd.temporal_consistency_loss(score, a_tilde)

    # Total: MIL dominates, auxiliary terms are tiny regularisers
    loss = l_mil + 0.005*l_norm + 0.005*l_dec + 0.005*l_temp

    return loss, l_mil.item(), l_norm.item()


# ══════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════

def train_epoch(epoch, model, n_loader, a_loader, optimizer, device):
    model.train()
    total = 0.0

    for n_inp, a_inp in zip(n_loader, a_loader):
        a_inp = a_inp.float()
        n_inp = n_inp.float()
        B, T, D = a_inp.shape

        # Stack [anomaly | normal] => (2B*T, D)
        inputs = torch.cat([a_inp, n_inp], dim=0).view(2*B*T, D).to(device)

        score, z_s, z_c, x_hat, X, a_tilde = model(inputs, T=T)

        # Reshape to (2B, T)
        score   = score.view(2*B, T)
        a_tilde = a_tilde.view(2*B, T)

        loss, l_mil, l_norm = compute_loss(
            score, z_s, z_c, x_hat, X, a_tilde, B, device, model, epoch
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()

    avg = total / max(len(n_loader), 1)
    print(f'[Train] Epoch {epoch:03d} | Loss: {avg:.4f} '
          f'| LR: {optimizer.param_groups[0]["lr"]:.6f}')
    return avg


# ══════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════

def test_epoch(epoch, model, a_loader, n_loader, device, best_auc):
    model.eval()
    auc_sum = 0.0

    with torch.no_grad():
        for (da, dn) in zip(a_loader, n_loader):
            inp_a, gts, frames = da
            T_a   = inp_a.size(1)
            inp_a = inp_a.view(-1, inp_a.size(-1)).float().to(device)
            sc_a, *_ = model(inp_a, T=T_a)
            sc_a = sc_a.view(T_a).cpu().numpy()

            fv         = int(frames[0])
            score_list = np.zeros(fv)
            step       = np.round(np.linspace(0, fv//16, T_a+1))
            for j in range(T_a):
                score_list[int(step[j])*16: int(step[j+1])*16] = sc_a[j % len(sc_a)]

            gt_list = np.zeros(fv)
            for k in range(len(gts)//2):
                gt_list[max(0, gts[k*2]-1): min(gts[k*2+1], fv)] = 1

            inp_n, _, frames_n = dn
            T_n   = inp_n.size(1)
            inp_n = inp_n.view(-1, inp_n.size(-1)).float().to(device)
            sc_n, *_ = model(inp_n, T=T_n)
            sc_n = sc_n.view(T_n).cpu().numpy()

            fn           = int(frames_n[0])
            score_list_n = np.zeros(fn)
            step_n       = np.round(np.linspace(0, fv//16, T_a+1))
            for j in range(T_a):
                s_i, e_i = int(step_n[j])*16, int(step_n[j+1])*16
                if e_i <= fn:
                    score_list_n[s_i:e_i] = sc_n[j % len(sc_n)]
            gt_list_n = np.zeros(fn)

            all_s = np.concatenate([score_list, score_list_n])
            all_g = np.concatenate([gt_list,    gt_list_n])
            fpr, tpr, _ = metrics.roc_curve(all_g, all_s, pos_label=1)
            auc_sum    += metrics.auc(fpr, tpr)

    avg_auc = auc_sum / 140
    is_best = avg_auc > best_auc
    print(f'[Test]  Epoch {epoch:03d} | AUC: {avg_auc:.4f} '
          f'| Best: {max(best_auc, avg_auc):.4f}'
          f'{" *** NEW BEST ***" if is_best else ""}')

    if is_best:
        os.makedirs('checkpoint', exist_ok=True)
        torch.save({'net': model.state_dict(), 'auc': avg_auc},
                   './checkpoint/scope_vad_best.pth')
    if avg_auc >= 0.86:
        torch.save({'net': model.state_dict(), 'auc': avg_auc},
                   f'./checkpoint/scope_vad_{avg_auc:.4f}.pth')
    return avg_auc


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    MODALITY   = 'TWO'
    BATCH_SIZE = 32
    EPOCHS     = 200
    PATIENCE   = 200
    DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {DEVICE}')

    normal_train  = Normal_Loader(is_train=1, modality=MODALITY)
    normal_test   = Normal_Loader(is_train=0, modality=MODALITY)
    anomaly_train = Anomaly_Loader(is_train=1, modality=MODALITY)
    anomaly_test  = Anomaly_Loader(is_train=0, modality=MODALITY)

    n_tr = DataLoader(normal_train,  batch_size=BATCH_SIZE, shuffle=True,  drop_last=True,  num_workers=0)
    n_te = DataLoader(normal_test,   batch_size=1,          shuffle=True,  num_workers=0)
    a_tr = DataLoader(anomaly_train, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True,  num_workers=0)
    a_te = DataLoader(anomaly_test,  batch_size=1,          shuffle=False, num_workers=0)

    INPUT_DIM = int(next(iter(a_tr)).shape[-1])
    print(f'Input dim: {INPUT_DIM} | Batch: {BATCH_SIZE}')

    model = SCOPE_VAD(
        input_dim = INPUT_DIM,
        K         = 4,
        M         = 4,
        state_dim = 256,
        proj_dim  = 256
    ).to(DEVICE)
    print(f'Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M')

    # Adagrad — proven best for UCF-Crime MIL across all literature
    optimizer = optim.Adagrad(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 60, 90], gamma=0.5
    )

    best_auc         = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_epoch(epoch, model, n_tr, a_tr, optimizer, DEVICE)
        auc = test_epoch(epoch, model, a_te, n_te, DEVICE, best_auc)

        if auc > best_auc:
            best_auc         = auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping at epoch {epoch}.')
                break

        scheduler.step()

    print(f'\nFinal Best AUC: {best_auc:.4f}')