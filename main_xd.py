"""
SCNP + SCSEC + OVCD on top of VadCLIP backbone
XD-Violence dataset — Target AP >= 84%

XD-Violence uses AP (Average Precision) as main metric, not AUC.
Categories: normal(A), fighting(B1), shooting(B2), riot(B4), abuse(B5),
            car accident(B6), explosion(G)
"""

import os, random, warnings
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.spatial.distance import pdist, squareform
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
class Cfg:
    TRAIN_FEAT    = r"C:\Users\khanm\Desktop\lab_project\Open_vocab\XDClipFeatures\XDTrainClipFeatures"
    TEST_FEAT     = r"C:\Users\khanm\Desktop\lab_project\Open_vocab\XDClipFeatures\XDTestClipFeatures"
    EXTRA_DIR     = r"C:\Users\khanm\Desktop\lab_project\Open_vocab\xd_extra"
    TRAIN_CSV     = os.path.join(EXTRA_DIR, "xd_CLIP_rgb.csv")
    TEST_CSV      = os.path.join(EXTRA_DIR, "xd_CLIP_rgbtest.csv")
    GT_NPY        = os.path.join(EXTRA_DIR, "gt.npy")
    GT_SEG_NPY    = os.path.join(EXTRA_DIR, "gt_segment.npy")
    GT_LABEL_NPY  = os.path.join(EXTRA_DIR, "gt_label.npy")

    FEAT_DIM      = 512
    T             = 256
    REPEAT        = 16

    # VadCLIP XD backbone params (from xd_option)
    EMBED_DIM     = 512
    VISUAL_WIDTH  = 512
    VISUAL_HEAD   = 1
    VISUAL_LAYERS = 1          # XD uses 1 layer (vs UCF=2)
    ATTN_WINDOW   = 64         # XD uses 64 (vs UCF=8)
    PROMPT_PREFIX = 10
    PROMPT_POSTFIX= 10
    CLASSES_NUM   = 7          # XD has 7 classes

    # Our modules
    K             = 8
    M             = 16
    PROJ_DIM      = 512

    EPOCHS        = 10
    BATCH_SIZE    = 96         # XD uses 96
    LR            = 1e-5       # XD uses 1e-5
    SCHEDULER_MILESTONES = [3, 6, 10]
    SCHEDULER_RATE       = 0.1
    AUX_W         = 0.005
    OVCD_W        = 0.01
    SEED          = 234
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

    # XD-Violence label map
    LABEL_MAP = {
        'A': 'normal', 'B1': 'fighting', 'B2': 'shooting',
        'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'
    }
    CATEGORIES = ['A', 'B1', 'B2', 'B4', 'B5', 'B6', 'G']
    ANOMALY_CATS = set(['B1','B2','B4','B5','B6','G'])

    EVENT_PROMPTS = [
        "a person running away quickly",
        "a sudden crowd gathering",
        "a person falling to the ground",
        "aggressive physical contact between people",
        "a person carrying suspicious objects",
        "vehicles moving erratically",
        "a person loitering suspiciously",
        "fire or smoke appearing suddenly",
        "a person screaming or in distress",
        "breaking glass or property damage",
        "a person being chased",
        "weapons or dangerous objects visible",
        "abnormal crowd behavior or panic",
        "a vehicle driving dangerously",
        "a person trespassing restricted area",
        "physical confrontation between multiple people",
    ]

cfg = Cfg()
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
random.seed(cfg.SEED)

# ══════════════════════════════════════════════════════════════
# VADCLIP LAYERS
# ══════════════════════════════════════════════════════════════
class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        orig = x.dtype
        return super().forward(x.float()).type(orig)

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()
        self.attn      = nn.MultiheadAttention(d_model, n_head)
        self.ln_1      = LayerNorm(d_model)
        self.mlp       = nn.Sequential(OrderedDict([
            ("c_fc",   nn.Linear(d_model, d_model*4)),
            ("gelu",   QuickGELU()),
            ("c_proj", nn.Linear(d_model*4, d_model))
        ]))
        self.ln_2      = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x, padding_mask):
        pm = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        am = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False,
                         key_padding_mask=pm, attn_mask=am)[0]

    def forward(self, x):
        x, pm = x
        x = x + self.attention(self.ln_1(x), pm)
        x = x + self.mlp(self.ln_2(x))
        return (x, pm)

class Transformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])
    def forward(self, x):
        return self.resblocks(x)

class GraphConvolution(Module):
    def __init__(self, in_f, out_f, bias=False, residual=True):
        super().__init__()
        self.in_features  = in_f
        self.out_features = out_f
        self.weight = Parameter(torch.FloatTensor(in_f, out_f))
        self.bias   = Parameter(torch.FloatTensor(out_f)) if bias else None
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None: self.bias.data.fill_(0.1)
        if not residual:
            self.residual = lambda x: 0
        elif in_f == out_f:
            self.residual = lambda x: x
        else:
            self.residual = nn.Conv1d(in_f, out_f, kernel_size=5, padding=2)

    def forward(self, inp, adj):
        out = adj.matmul(inp.matmul(self.weight))
        if self.bias is not None: out = out + self.bias
        if self.in_features != self.out_features and callable(self.residual):
            res = self.residual(inp.permute(0,2,1)).permute(0,2,1)
            out = out + res
        else:
            out = out + self.residual(inp)
        return out

class DistanceAdj(Module):
    def __init__(self):
        super().__init__()
        self.sigma = Parameter(torch.FloatTensor(1))
        self.sigma.data.fill_(0.1)

    def forward(self, batch_size, max_seqlen):
        arith = np.arange(max_seqlen).reshape(-1,1)
        dist  = pdist(arith, metric='cityblock').astype(np.float32)
        dist  = torch.from_numpy(squareform(dist)).to(cfg.DEVICE)
        dist  = torch.exp(-dist / torch.exp(torch.tensor(1.)))
        return dist.unsqueeze(0).repeat(batch_size,1,1)

# ══════════════════════════════════════════════════════════════
# CLIP TEXT FEATURES
# ══════════════════════════════════════════════════════════════
def get_clip_text_features(prompts, device):
    try:
        import clip as clip_lib
        print("  Loading CLIP text encoder (ViT-B/32)...")
        m, _ = clip_lib.load("ViT-B/32", device=device)
        m.eval()
        with torch.no_grad():
            t = clip_lib.tokenize(prompts).to(device)
            f = F.normalize(m.encode_text(t).float(), dim=-1)
        del m; torch.cuda.empty_cache()
        return f.detach()
    except Exception as e:
        print(f"  CLIP error: {e}")
        return F.normalize(torch.randn(len(prompts), cfg.FEAT_DIM), dim=-1).to(device)

def get_prompt_text(label_map):
    return list(label_map.values())

# ══════════════════════════════════════════════════════════════
# FEATURE PROCESSING
# ══════════════════════════════════════════════════════════════
def uniform_extract(feat, t_max):
    new_feat = np.zeros((t_max, feat.shape[1]), np.float32)
    r = np.linspace(0, len(feat), t_max+1, dtype=np.int32)
    for i in range(t_max):
        new_feat[i] = np.mean(feat[r[i]:r[i+1]], 0) if r[i]!=r[i+1] else feat[r[i]]
    return new_feat

def pad_feat(feat, min_len):
    if feat.shape[0] <= min_len:
        return np.pad(feat,((0,min_len-feat.shape[0]),(0,0)),
                      mode='constant',constant_values=0)
    return feat

def process_feat(feat, length):
    if feat.shape[0] > length: return uniform_extract(feat, length), length
    return pad_feat(feat, length), feat.shape[0]

def process_split(feat, length):
    n = feat.shape[0]
    if n < length: return pad_feat(feat,length)[np.newaxis], n
    chunks = []
    for i in range(int(n/length)+1):
        chunks.append(pad_feat(feat[i*length:(i+1)*length], length)[np.newaxis])
    return np.concatenate(chunks,0), n

def fix_path_xd(p, is_test=False):
    """Fix Linux path to Windows path for XD dataset."""
    basename = os.path.basename(p)
    folder   = cfg.TEST_FEAT if is_test else cfg.TRAIN_FEAT
    return os.path.join(folder, basename)

# ══════════════════════════════════════════════════════════════
# DATASET — XD TRAIN
# ══════════════════════════════════════════════════════════════
class XDTrainDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, sep=None, engine="python")
        df.columns  = [c.strip() for c in df.columns]
        df["label"] = df["label"].str.strip()
        df["fpath"] = df["path"].apply(lambda p: fix_path_xd(p, is_test=False))
        self.df = df
        # Count normal vs anomaly
        n_normal = len(df[df["label"]=="A"])
        n_anom   = len(df[df["label"]!="A"])
        print(f"  XD Train — normal:{n_normal} anomaly:{n_anom} total:{len(df)}")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            feat = np.load(row["fpath"]).astype(np.float32)
            if feat.ndim==1: feat=feat[np.newaxis]
        except:
            feat = np.zeros((1,cfg.FEAT_DIM),np.float32)
        feat, length = process_feat(feat, cfg.T)
        label   = row["label"]
        cat_idx = cfg.CATEGORIES.index(label) if label in cfg.CATEGORIES else 0
        return torch.tensor(feat), label, torch.tensor(length), torch.tensor(cat_idx)

# ══════════════════════════════════════════════════════════════
# DATASET — XD TEST
# ══════════════════════════════════════════════════════════════
class XDTestDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, sep=None, engine="python")
        df.columns  = [c.strip() for c in df.columns]
        df["label"] = df["label"].str.strip()
        df["fpath"] = df["path"].apply(lambda p: fix_path_xd(p, is_test=True))
        self.df = df
        print(f"  XD Test clips: {len(df)}")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            feat = np.load(row["fpath"]).astype(np.float32)
            if feat.ndim==1: feat=feat[np.newaxis]
        except:
            feat = np.zeros((1,cfg.FEAT_DIM),np.float32)
        feat, length = process_split(feat, cfg.T)
        return torch.tensor(feat), row["label"], torch.tensor(length)

# ══════════════════════════════════════════════════════════════
# MODULE 1 — SCNP
# ══════════════════════════════════════════════════════════════
class SCNP(nn.Module):
    def __init__(self, D, K):
        super().__init__()
        self.K = K
        self.omega_net = nn.Sequential(
            nn.Linear(D,128), nn.ReLU(), nn.Linear(128,1), nn.Sigmoid()
        )
        self.proto_W = nn.Linear(D, D*K)
        nn.init.xavier_uniform_(self.proto_W.weight)

    def forward(self, X):
        B,T,D = X.shape
        omega = self.omega_net(X).squeeze(-1)
        w     = omega/(omega.sum(1,keepdim=True)+1e-8)
        s     = (w.unsqueeze(-1)*X).sum(1)
        P     = self.proto_W(s).view(B,self.K,D)
        dist  = -((X.unsqueeze(2)-P.unsqueeze(1))**2).sum(-1)/D**0.5
        alpha = torch.softmax(dist,dim=-1)
        x_hat = torch.einsum('btk,bkd->btd',alpha,P)
        a_t   = torch.sigmoid(torch.norm(X-x_hat,dim=-1))
        return x_hat, a_t, s, omega, P

    def aux_loss(self, X, x_hat, omega, P):
        recon = (omega*((X-x_hat)**2).sum(-1)).mean()
        p_n   = F.normalize(P,dim=-1)
        sim   = torch.bmm(p_n,p_n.transpose(1,2))
        mask  = 1-torch.eye(self.K,device=P.device).unsqueeze(0)
        div   = (sim*mask).sum()/(P.shape[0]*self.K*(self.K-1)+1e-8)
        return recon + 0.1*div

# ══════════════════════════════════════════════════════════════
# MODULE 2 — SCSEC
# ══════════════════════════════════════════════════════════════
class SCSEC(nn.Module):
    def __init__(self, D, M, clip_event_feats, lam=0.5):
        super().__init__()
        self.M=M; self.lam=lam
        self.register_buffer("clip_event_init", clip_event_feats)
        self.event_scale = nn.Parameter(torch.ones(M,1))
        self.event_bias  = nn.Parameter(torch.zeros(M,cfg.FEAT_DIM))
        self.phi = nn.Sequential(
            nn.Linear(D,cfg.FEAT_DIM), nn.LayerNorm(cfg.FEAT_DIM),
            nn.ReLU(), nn.Linear(cfg.FEAT_DIM,cfg.FEAT_DIM)
        )
        self.scene_proj  = nn.Linear(D,cfg.FEAT_DIM)
        self.scene_adapt = nn.Linear(cfg.FEAT_DIM*2,cfg.FEAT_DIM)
        self.trans_pred  = nn.Sequential(
            nn.Linear(cfg.FEAT_DIM*2,cfg.FEAT_DIM),
            nn.ReLU(), nn.Linear(cfg.FEAT_DIM,cfg.FEAT_DIM)
        )

    def get_event_protos(self):
        return F.normalize(
            self.clip_event_init*self.event_scale+self.event_bias, dim=-1
        )

    def forward(self, x_resid, s):
        B,T,D = x_resid.shape
        v  = self.phi(x_resid)
        s_ = self.scene_proj(s)
        b  = self.get_event_protos()
        b_s= F.normalize(self.scene_adapt(
            torch.cat([s_.unsqueeze(1).expand(-1,self.M,-1),
                       b.unsqueeze(0).expand(B,-1,-1)],dim=-1)), dim=-1)
        gamma   = F.softmax(torch.bmm(F.normalize(v,dim=-1),b_s.transpose(1,2)),dim=-1)
        c       = torch.bmm(gamma,b_s)
        c_prev  = torch.cat([c[:,:1],c[:,:-1]],dim=1)
        d_obs   = c-c_prev
        d_exp   = self.trans_pred(
            torch.cat([c_prev,s_.unsqueeze(1).expand(-1,T,-1)],dim=-1))
        e_t     = torch.norm(d_exp-d_obs,dim=-1)
        a_sem   = 1.0-torch.bmm(
            F.normalize(c,dim=-1),b_s.transpose(1,2)).max(-1).values
        a_tilde = torch.sigmoid(e_t+self.lam*a_sem)
        return c, a_tilde, d_exp, d_obs

    def aux_loss(self, d_exp, d_obs, omega):
        return (omega*((d_exp-d_obs)**2).sum(-1)).mean()

# ══════════════════════════════════════════════════════════════
# MODULE 3 — OVCD
# ══════════════════════════════════════════════════════════════
class OVCD(nn.Module):
    def __init__(self, D, C, vocab_feats):
        super().__init__()
        self.C = C
        self.register_buffer("vocab", vocab_feats)
        self.f_s  = nn.Sequential(
            nn.Linear(D,cfg.PROJ_DIM), nn.LayerNorm(cfg.PROJ_DIM), nn.ReLU()
        )
        self.f_c  = nn.Sequential(
            nn.Linear(cfg.FEAT_DIM,cfg.PROJ_DIM),
            nn.LayerNorm(cfg.PROJ_DIM), nn.ReLU()
        )
        self.gate  = nn.Linear(cfg.PROJ_DIM*2, cfg.PROJ_DIM)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x_hat, c, a_t, a_tilde):
        z_s  = self.f_s(x_hat)
        z_c  = self.f_c(c)
        beta = torch.sigmoid(self.gate(torch.cat([z_s,z_c],dim=-1)))
        z    = beta*z_s + (1-beta)*z_c
        z_n  = F.normalize(z, dim=-1)
        v_n  = F.normalize(self.vocab, dim=-1)
        s_tc = torch.matmul(z_n, v_n.T)
        alp  = torch.sigmoid(self.alpha)
        sem  = 1.0 - s_tc.max(dim=-1).values
        A_t  = alp*sem + (1-alp)*a_tilde
        return z, z_s, z_c, s_tc, A_t

    def decouple_loss(self, z_s, z_c):
        zs = F.normalize(z_s.reshape(-1,z_s.size(-1)),dim=-1)
        zc = F.normalize(z_c.reshape(-1,z_c.size(-1)),dim=-1)
        return ((zs*zc).sum(-1)**2).mean()

    def temp_loss(self, s_tc, a_tilde):
        conf = torch.sigmoid(s_tc.max(dim=-1).values)
        return F.mse_loss(conf, a_tilde.detach().clamp(0,1))

    def scn_loss(self, s_tc, cat_idx):
        B,T,C  = s_tc.shape
        s_mean = s_tc.mean(dim=1)
        pos    = s_mean[torch.arange(B), cat_idx]
        scene_idx = list(range(1, C))
        return (-pos + torch.logsumexp(s_mean[:,scene_idx], dim=-1)).mean()

    def ov_loss(self, s_tc, cat_idx):
        B,T,C  = s_tc.shape
        s_mean = s_tc.mean(dim=1)
        pos    = s_mean[torch.arange(B), cat_idx]
        return (-pos + torch.logsumexp(s_mean, dim=-1)).mean()

# ══════════════════════════════════════════════════════════════
# FULL MODEL
# ══════════════════════════════════════════════════════════════
class VADModel(nn.Module):
    def __init__(self, clip_event_feats, clip_vocab_feats):
        super().__init__()
        W  = cfg.VISUAL_WIDTH
        hw = W // 2

        self.temporal = Transformer(
            width=W, layers=cfg.VISUAL_LAYERS, heads=cfg.VISUAL_HEAD,
            attn_mask=self._build_attn_mask(cfg.ATTN_WINDOW)
        )
        self.gc1    = GraphConvolution(W, hw, residual=True)
        self.gc2    = GraphConvolution(hw, hw, residual=True)
        self.gc3    = GraphConvolution(W, hw, residual=True)
        self.gc4    = GraphConvolution(hw, hw, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(W, W)
        self.gelu   = QuickGELU()
        self.mlp1   = nn.Sequential(OrderedDict([
            ("c_fc",nn.Linear(W,W*4)),("gelu",QuickGELU()),("c_proj",nn.Linear(W*4,W))
        ]))
        self.mlp2   = nn.Sequential(OrderedDict([
            ("c_fc",nn.Linear(W,W*4)),("gelu",QuickGELU()),("c_proj",nn.Linear(W*4,W))
        ]))
        self.classifier = nn.Linear(W, 1)
        self.pos_embed  = nn.Embedding(cfg.T, W)
        nn.init.normal_(self.pos_embed.weight, std=0.01)

        import clip as clip_lib
        self.clipmodel, _ = clip_lib.load("ViT-B/16", cfg.DEVICE)
        for p in self.clipmodel.parameters():
            p.requires_grad = False

        self.scnp  = SCNP(W, cfg.K)
        self.scsec = SCSEC(W, cfg.M, clip_event_feats)
        self.ovcd  = OVCD(W, cfg.CLASSES_NUM, clip_vocab_feats)

    def _build_attn_mask(self, attn_window):
        mask = torch.empty(cfg.T, cfg.T).fill_(float('-inf'))
        for i in range(int(cfg.T/attn_window)):
            s = i*attn_window
            e = min(s+attn_window, cfg.T)
            mask[s:e, s:e] = 0
        return mask

    def _adj4(self, x, lengths):
        x2     = x.matmul(x.permute(0,2,1))
        x_norm = torch.norm(x,p=2,dim=2,keepdim=True)
        x2     = x2/(x_norm.matmul(x_norm.permute(0,2,1))+1e-20)
        output = torch.zeros_like(x2)
        for i in range(len(lengths)):
            tmp = x2[i,:lengths[i],:lengths[i]]
            tmp = F.threshold(tmp,0.7,0)
            tmp = F.softmax(tmp,dim=1)
            output[i,:lengths[i],:lengths[i]] = tmp
        return output

    def encode_video(self, x, lengths):
        x   = x.float()
        pos = self.pos_embed(
            torch.arange(cfg.T,device=x.device).unsqueeze(0).expand(x.shape[0],-1)
        ).permute(1,0,2)
        x   = x.permute(1,0,2) + pos
        x,_ = self.temporal((x, None))
        x   = x.permute(1,0,2)
        adj    = self._adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1 = self.gelu(self.gc2(self.gelu(self.gc1(x,adj)),adj))
        x2 = self.gelu(self.gc4(self.gelu(self.gc3(x,disadj)),disadj))
        return self.linear(torch.cat([x1,x2],2))

    def encode_text(self, text):
        import clip as clip_lib
        with torch.no_grad():
            tokens = clip_lib.tokenize(text).to(cfg.DEVICE)
            feats  = self.clipmodel.encode_text(tokens).float()
            feats  = F.normalize(feats, dim=-1)
        return feats

    def forward(self, visual, lengths, prompt_text):
        vf = self.encode_video(visual, lengths)
        x_hat, a_t, s, omega, P = self.scnp(vf)
        x_resid = (vf - x_hat).detach()
        c, a_tilde, d_exp, d_obs = self.scsec(x_resid, s)
        z, z_s, z_c, s_tc, A_t  = self.ovcd(x_hat, c, a_t, a_tilde)
        vf_enhanced = vf + z
        logits1 = self.classifier(vf_enhanced + self.mlp2(vf_enhanced))
        text_feat_ori = self.encode_text(prompt_text)
        logits_attn   = logits1.permute(0,2,1)
        v_attn = logits_attn @ vf_enhanced
        v_attn = v_attn / (v_attn.norm(dim=-1,keepdim=True)+1e-8)
        v_attn = v_attn.expand(-1, text_feat_ori.shape[0], -1)
        tf = text_feat_ori.unsqueeze(0).expand(v_attn.shape[0],-1,-1)
        tf = tf + v_attn + self.mlp1(tf + v_attn)
        vf_norm = vf_enhanced / (vf_enhanced.norm(dim=-1,keepdim=True)+1e-8)
        tf_norm = (tf / (tf.norm(dim=-1,keepdim=True)+1e-8)).permute(0,2,1)
        logits2 = vf_norm @ tf_norm.type(vf_norm.dtype) / 0.07
        return (text_feat_ori, logits1, logits2,
                x_hat, vf, omega, P,
                a_tilde, d_exp, d_obs,
                z_s, z_c, s_tc, A_t)

# ══════════════════════════════════════════════════════════════
# LOSSES
# ══════════════════════════════════════════════════════════════
def CLAS2_loss(logits1, labels_bin, lengths, device):
    B     = logits1.shape[0]
    probs = torch.sigmoid(logits1).reshape(B,-1)
    inst  = torch.zeros(0).to(device)
    for i in range(B):
        k   = max(1, min(int(int(lengths[i])/16+1), int(lengths[i])))
        tmp = probs[i,:int(lengths[i])].topk(k).values.mean().view(1)
        inst= torch.cat([inst,tmp])
    return F.binary_cross_entropy(inst, labels_bin.float().to(device))

def CLASM_loss(logits2, labels_cls, lengths, device):
    B    = logits2.shape[0]
    inst = torch.zeros(0).to(device)
    lbl  = (labels_cls/(labels_cls.sum(1,keepdim=True)+1e-8)).to(device)
    for i in range(B):
        k   = max(1, min(int(int(lengths[i])/16+1), int(lengths[i])))
        tmp,_ = logits2[i,:int(lengths[i])].topk(k,dim=0,largest=True)
        inst  = torch.cat([inst,tmp.mean(0,keepdim=True)],dim=0)
    return -torch.mean(torch.sum(lbl*F.log_softmax(inst,dim=1),dim=1))

def text_diversity_loss(text_feats, device):
    tf   = text_feats / (text_feats.norm(dim=-1,keepdim=True)+1e-8)
    loss = torch.tensor(0.).to(device)
    n    = tf.shape[0]
    for j in range(1, n):
        loss += torch.abs(tf[0] @ tf[j])
    return loss / max(n-1,1) * 1e-4

def get_label_vectors(labels):
    C    = len(cfg.CATEGORIES)
    vecs = []
    for l in labels:
        v = torch.zeros(C)
        if l in cfg.CATEGORIES: v[cfg.CATEGORIES.index(l)] = 1
        vecs.append(v)
    return torch.stack(vecs)

# ══════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════
def train_epoch(epoch, model, train_loader, optimizer, device,
                test_loader, best_ap, gt, prompt_text):
    model.train()
    l1_tot = l2_tot = 0.0
    steps  = 0

    for nf,nl,nlen,ncat in train_loader:
        visual  = nf.float().to(device)
        lengths = nlen
        labels  = list(nl)
        B       = visual.shape[0]

        out = model(visual, lengths, prompt_text)
        text_feats,logits1,logits2,x_hat,vf,omega,P,\
            a_tilde,d_exp,d_obs,z_s,z_c,s_tc,A_t = out

        labels_cls = get_label_vectors(labels)
        labels_bin = torch.tensor([0. if l=="A" else 1. for l in labels])
        cat_idx    = torch.tensor([
            cfg.CATEGORIES.index(l) if l in cfg.CATEGORIES else 0
            for l in labels
        ]).to(device)

        l1 = CLAS2_loss(logits1, labels_bin, lengths, device)
        l2 = CLASM_loss(logits2, labels_cls, lengths, device)
        l3 = text_diversity_loss(text_feats, device)

        # Normal bags for SCNP/SCSEC aux
        norm_idx = [i for i,l in enumerate(labels) if l=="A"]
        if len(norm_idx) > 0:
            ni = torch.tensor(norm_idx)
            l_scnp  = model.scnp.aux_loss(
                vf[ni], x_hat[ni], omega[ni], P[ni])
            l_scsec = model.scsec.aux_loss(
                d_exp[ni], d_obs[ni], omega[ni])
        else:
            l_scnp  = torch.tensor(0.).to(device)
            l_scsec = torch.tensor(0.).to(device)

        l_dec  = model.ovcd.decouple_loss(z_s, z_c)
        l_temp = model.ovcd.temp_loss(s_tc, a_tilde)
        l_scn  = model.ovcd.scn_loss(s_tc, cat_idx)
        l_ov   = model.ovcd.ov_loss(s_tc, cat_idx)

        loss = (l1 + l2 + l3
                + cfg.AUX_W  * (l_scnp + l_scsec)
                + cfg.OVCD_W * (l_dec + l_temp + l_scn + l_ov))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        l1_tot += l1.item()
        l2_tot += l2.item()
        steps  += 1

        step = steps * cfg.BATCH_SIZE
        if step % 4800 == 0 and step != 0:
            print(f"  epoch:{epoch} step:{step} "
                  f"loss1:{l1_tot/steps:.4f} loss2:{l2_tot/steps:.4f}")
            ap, auc = evaluate(epoch, model, test_loader, device,
                               best_ap, gt, prompt_text)
            if ap > best_ap: best_ap = ap
            model.train()

    print(f"[Train] Epoch {epoch:03d} | "
          f"loss1:{l1_tot/max(steps,1):.4f} "
          f"loss2:{l2_tot/max(steps,1):.4f} "
          f"LR:{optimizer.param_groups[0]['lr']:.2e}")
    return best_ap

# ══════════════════════════════════════════════════════════════
# EVALUATE — XD uses AP as main metric
# ══════════════════════════════════════════════════════════════
def evaluate(epoch, model, test_loader, device, best_ap, gt, prompt_text):
    model.eval()
    gt_bin = (gt > 0).astype(int)

    ap1_all = []
    ap2_all = []

    with torch.no_grad():
        for feat, label, length in test_loader:
            feat    = feat.squeeze(0).float().to(device)
            len_cur = int(length[0])
            if feat.ndim==2: feat=feat.unsqueeze(0)

            sc1_list=[]; sc2_list=[]
            for ci in range(feat.shape[0]):
                chunk = feat[ci:ci+1]
                ln    = torch.tensor([min(cfg.T, max(1, len_cur-ci*cfg.T))])
                out   = model(chunk, ln, prompt_text)
                _,l1,l2 = out[0],out[1],out[2]
                sc1_list.append(torch.sigmoid(l1).reshape(-1).cpu().numpy())
                sc2_list.append((1-l2.softmax(-1)[...,0]).reshape(-1).cpu().numpy())

            sc1 = np.concatenate(sc1_list)[:len_cur]
            sc2 = np.concatenate(sc2_list)[:len_cur]
            ap1_all.append(sc1)
            ap2_all.append(sc2)

    ap1 = np.repeat(np.concatenate(ap1_all), cfg.REPEAT)
    ap2 = np.repeat(np.concatenate(ap2_all), cfg.REPEAT)

    def match(arr, n):
        if len(arr)>n: return arr[:n]
        return np.pad(arr,(0,n-len(arr)),mode='edge')

    ap1 = match(ap1, len(gt_bin))
    ap2 = match(ap2, len(gt_bin))

    auc1 = roc_auc_score(gt_bin, ap1)
    ap1v = average_precision_score(gt_bin, ap1)
    auc2 = roc_auc_score(gt_bin, ap2)
    ap2v = average_precision_score(gt_bin, ap2)

    best_ap_now = max(ap1v, ap2v)
    is_best     = best_ap_now > best_ap
    marker      = " *** NEW BEST ***" if is_best else ""

    print(f"[Test]  Epoch {epoch:03d} | "
          f"AUC1:{auc1:.4f} AP1:{ap1v:.4f} | "
          f"AUC2:{auc2:.4f} AP2:{ap2v:.4f} | "
          f"Best AP:{max(best_ap,best_ap_now):.4f}{marker}")

    if is_best:
        os.makedirs("checkpoint", exist_ok=True)
        torch.save({"net": model.state_dict(), "ap": best_ap_now},
                   "checkpoint/best_model_xd.pth")
    return best_ap_now, max(auc1, auc2)

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print(f"Device: {cfg.DEVICE}")

    print("Loading CLIP features...")
    clip_event_feats = get_clip_text_features(cfg.EVENT_PROMPTS, cfg.DEVICE)
    clip_vocab_feats = get_clip_text_features(
        list(cfg.LABEL_MAP.values()), cfg.DEVICE
    )
    print(f"  Events:{clip_event_feats.shape} Vocab:{clip_vocab_feats.shape}")

    print("Loading datasets...")
    train_ds  = XDTrainDataset(cfg.TRAIN_CSV)
    test_ds   = XDTestDataset(cfg.TEST_CSV)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True, drop_last=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=False, num_workers=0)

    gt = np.load(cfg.GT_NPY)
    prompt_text = get_prompt_text(cfg.LABEL_MAP)

    model    = VADModel(clip_event_feats, clip_vocab_feats).to(cfg.DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params/1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR)
    scheduler = MultiStepLR(optimizer,
                            milestones=cfg.SCHEDULER_MILESTONES,
                            gamma=cfg.SCHEDULER_RATE)

    best_ap = 0.0
    for epoch in range(1, cfg.EPOCHS+1):
        best_ap = train_epoch(epoch, model, train_loader,
                              optimizer, cfg.DEVICE,
                              test_loader, best_ap, gt, prompt_text)
        ap, auc = evaluate(epoch, model, test_loader, cfg.DEVICE,
                           best_ap, gt, prompt_text)
        if ap > best_ap: best_ap = ap
        scheduler.step()

    print(f"\nFinal Best AP: {best_ap:.4f}")
    print("Best model saved to checkpoint/best_model_xd.pth")

if __name__ == "__main__":
    main()