import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    from Bio.SeqUtils import gc_fraction
    _HAS_BIO = True
except ImportError:
    _HAS_BIO = False

try:
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, rdMolDescriptors, AllChem
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False


def normalize_sym(adj):
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_row(mx):
    rowsum = np.array(mx.sum(1)).flatten()
    r_inv = np.power(rowsum, -1.0)
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx).tocoo()


def sparse_to_torch(sparse_mx):
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data.astype(np.float32))
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def _read_excel(path, **kwargs):
    if not _HAS_PANDAS:
        raise ImportError(
            "pandas + openpyxl are required to read Excel data files. "
            "Install with: pip install pandas openpyxl")
    return pd.read_excel(path, **kwargs)


def load_associations(data_dir):
    mirna_name = []
    lncRNA_name = []
    drug_name = []

    def _append(name_list, name):
        if name not in name_list:
            name_list.append(name)
        return name_list.index(name)

    md = _read_excel(
        os.path.join(data_dir, 'miRNA_drug_associations.xlsx'),
        header=None, names=['mirna', 'drug'])
    md['mirna'] = md['mirna'].str.lower().str.strip()
    md['drug'] = md['drug'].str.lower().str.strip()
    md_src, md_dst = [], []
    for _, row in md.iterrows():
        mi = _append(mirna_name, row['mirna'])
        dr = _append(drug_name, row['drug'])
        md_src.append(mi)
        md_dst.append(dr)
    md_edges = np.array([md_src, md_dst], dtype=np.int64)

    ld = _read_excel(
        os.path.join(data_dir, 'lncRNA_drug_associations.xlsx'),
        header=None, names=['lncRNA', 'drug'])
    ld['lncRNA'] = ld['lncRNA'].str.lower().str.strip()
    ld['drug'] = ld['drug'].str.lower().str.strip()
    ld_src, ld_dst = [], []
    for _, row in ld.iterrows():
        ln = _append(lncRNA_name, row['lncRNA'])
        dr = _append(drug_name, row['drug'])
        ld_src.append(ln)
        ld_dst.append(dr)
    ld_edges = np.array([ld_src, ld_dst], dtype=np.int64)

    ml = _read_excel(
        os.path.join(data_dir, 'miRNA_lncRNA_associations.xlsx'),
        header=None, names=['lncRNA', 'mirna'])
    ml['mirna'] = ml['mirna'].str.lower().str.strip()
    ml['lncRNA'] = ml['lncRNA'].str.lower().str.strip()
    ml_src, ml_dst = [], []
    for _, row in ml.iterrows():
        mi = _append(mirna_name, row['mirna'])
        ln = _append(lncRNA_name, row['lncRNA'])
        ml_src.append(mi)
        ml_dst.append(ln)
    ml_edges = np.array([ml_src, ml_dst], dtype=np.int64)

    print(f"[Data] miRNA={len(mirna_name)}, "
          f"lncRNA={len(lncRNA_name)}, drug={len(drug_name)}")
    print(f"[Data] edges: miRNA-drug={md_edges.shape[1]}, "
          f"lncRNA-drug={ld_edges.shape[1]}, miRNA-lncRNA={ml_edges.shape[1]}")

    return mirna_name, lncRNA_name, drug_name, md_edges, ld_edges, ml_edges


def _kmer_permutations(letters, k):
    if k == 1:
        return list(letters)
    prev = _kmer_permutations(letters, k - 1)
    return [p + c for p in prev for c in letters]


def _kmer_features(sequences, letters, k=3):
    vocab = _kmer_permutations(letters, k)
    vocab_idx = {v: i for i, v in enumerate(vocab)}
    feats = np.zeros((len(sequences), len(vocab)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        if not seq:
            continue
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j + k]
            if kmer in vocab_idx:
                feats[i, vocab_idx[kmer]] += 1
        total = feats[i].sum()
        if total > 0:
            feats[i] /= total
    return feats


def _secondary_structure_features(sequences, is_rna=True):
    from collections import Counter
    import math
    base_pairs = {'A': 'U', 'G': 'C', 'C': 'G', 'U': 'A'} if is_rna \
        else {'A': 'T', 'G': 'C', 'C': 'G', 'T': 'A'}
    feats = np.zeros((len(sequences), 4), dtype=np.float32)
    for i, seq in enumerate(sequences):
        if not seq:
            continue
        gc = sum(1 for b in seq if b in 'GC') / len(seq)
        windows = [seq[j:j + 5] for j in range(len(seq) - 4)] or [seq]
        local_gc = np.mean([sum(1 for b in w if b in 'GC') / len(w)
                            for w in windows])
        hb = 0
        for j in range(len(seq) - 1):
            if seq[j] in base_pairs and seq[j + 1] == base_pairs[seq[j]]:
                hb += 3 if seq[j] in 'GC' else 2
        kmers = [seq[j:j + 3] for j in range(len(seq) - 2)]
        if kmers:
            counts = Counter(kmers)
            total = len(kmers)
            entropy = -sum((c / total) * math.log2(c / total)
                           for c in counts.values() if c > 0)
        else:
            entropy = 0.0
        feats[i] = [gc, local_gc, hb, entropy]
    return feats


class _RNASeqFusion(nn.Module):
    def __init__(self, kmer_dim, sec_dim, sim_dim,
                 kmer_reduce=64, sec_reduce=8, sim_reduce=64, out=128):
        super().__init__()
        self.kmer_linear = nn.Linear(kmer_dim, kmer_reduce)
        self.sec_linear = nn.Linear(sec_dim, sec_reduce)
        self.sim_linear = nn.Linear(sim_dim, sim_reduce)
        self.alpha_kmer = nn.Parameter(torch.ones(kmer_reduce))
        self.alpha_sec = nn.Parameter(torch.ones(sec_reduce))
        self.alpha_sim = nn.Parameter(torch.ones(sim_reduce))
        self.transformer = nn.Linear(kmer_reduce + sec_reduce + sim_reduce, out)

    def forward(self, kmer, sec, sim):
        k = self.kmer_linear(kmer) * self.alpha_kmer
        s = self.sec_linear(sec) * self.alpha_sec
        m = self.sim_linear(sim) * self.alpha_sim
        return self.transformer(torch.cat([k, s, m], dim=1))


class _DrugFusion(nn.Module):
    def __init__(self, morgan_size=2048, maccs_size=167,
                 daylight_size=2048, out=128):
        super().__init__()
        self.morgan_fc = nn.Linear(morgan_size, 512)
        self.maccs_fc = nn.Linear(maccs_size, 512)
        self.daylight_fc = nn.Linear(daylight_size, 512)
        self.attention = nn.Linear(512 * 3, 1)
        self.fusion_fc = nn.Linear(512 * 3, out)
        self.relu = nn.ReLU()

    def forward(self, morgan, maccs, daylight):
        m = self.relu(self.morgan_fc(morgan))
        a = self.relu(self.maccs_fc(maccs))
        d = self.relu(self.daylight_fc(daylight))
        combined = torch.cat([m, a, d], dim=1)
        attn = torch.softmax(self.attention(combined), dim=1)
        return self.fusion_fc(combined * attn)


def _load_seq_map(path):
    df = _read_excel(path, header=None)
    df.columns = ['name', 'seq']
    return dict(zip(df['name'].str.lower().str.strip(), df['seq']))


_SIM_CACHE = {}


def _load_sim_matrix(path):
    abs_path = os.path.abspath(path)
    if abs_path in _SIM_CACHE:
        return _SIM_CACHE[abs_path]

    df = _read_excel(path, index_col=0)
    row_names = [str(n).lower().strip() for n in df.index]
    col_names = [str(c).lower().strip() for c in df.columns]
    mat = df.values.astype(np.float32)
    result = (row_names, col_names, mat)
    _SIM_CACHE[abs_path] = result
    print(f"[Data] loaded sim matrix: {os.path.basename(path)} "
          f"shape={mat.shape}, row_names={len(row_names)}, "
          f"col_names={len(col_names)}")
    return result


def _normalize_rna_name(name):
    import re
    s = name.lower().strip()
    s = re.sub(r'^(hsa|mmu|rno|dre|cel|dme)-', '', s)
    s = re.sub(r'\.\d+$', '', s)
    return s


def _build_name_index(names):
    idx = {}
    for i, n in enumerate(names):
        norm = _normalize_rna_name(n)
        idx.setdefault(norm, []).append(i)
    return idx


def _lookup_name_index(norm_name, name_index, sim_mat, axis='row'):
    import re
    if norm_name in name_index:
        indices = name_index[norm_name]
        if len(indices) == 1:
            return sim_mat[indices[0]] if axis == 'row' else sim_mat[:, indices[0]]
        sub = (sim_mat[indices] if axis == 'row'
               else sim_mat[:, indices])
        return sub.mean(axis=0 if axis == 'row' else 1)

    m = re.match(r'^(.+)-\d+$', norm_name)
    if m:
        base = m.group(1)
        if base in name_index:
            indices = name_index[base]
            sub = (sim_mat[indices] if axis == 'row'
                   else sim_mat[:, indices])
            return sub.mean(axis=0 if axis == 'row' else 1)
    return None


def _align_sim_matrix(sim_row_names, sim_col_names, sim_mat, target_names,
                      name=""):
    n = len(target_names)
    row_index = _build_name_index(sim_row_names)
    col_index = _build_name_index(sim_col_names)

    aligned = np.zeros((n, n), dtype=np.float32)
    row_hits = 0
    col_hits = 0

    target_rows = []
    for ni in target_names:
        norm = _normalize_rna_name(ni)
        vec = _lookup_name_index(norm, row_index, sim_mat, axis='row')
        if vec is not None:
            row_hits += 1
        target_rows.append(vec)

    target_cols = []
    for nj in target_names:
        norm = _normalize_rna_name(nj)
        vec = _lookup_name_index(norm, col_index, sim_mat, axis='col')
        if vec is not None:
            col_hits += 1
        target_cols.append(vec)

    for i in range(n):
        if target_rows[i] is None:
            continue
        for j in range(n):
            if target_cols[j] is None:
                continue
            norm_j = _normalize_rna_name(target_names[j])
            col_indices = col_index.get(norm_j)
            if col_indices is None:
                import re
                m = re.match(r'^(.+)-\d+$', norm_j)
                if m and m.group(1) in col_index:
                    col_indices = col_index[m.group(1)]
            if col_indices:
                aligned[i, j] = float(np.mean([target_rows[i][c]
                                                for c in col_indices]))

    print(f"[Data] {name} sim alignment: "
          f"row_match={row_hits}/{n} ({row_hits/n*100:.1f}%), "
          f"col_match={col_hits}/{n} ({col_hits/n*100:.1f}%)")
    if row_hits == 0 or col_hits == 0:
        print(f"[Data] WARNING: {name} sim matrix has zero matches! "
              f"Sample target names: {target_names[:5]}")
        print(f"[Data] Sample sim row names: {sim_row_names[:5]}")
        print(f"[Data] Sample sim col names: {sim_col_names[:5]}")
        print(f"[Data] Sample normalised target: "
              f"{[_normalize_rna_name(n) for n in target_names[:5]]}")
        print(f"[Data] Sample normalised sim rows: "
              f"{[_normalize_rna_name(n) for n in sim_row_names[:5]]}")
    return aligned


def extract_rna_features(names, seq_path, sim_path, is_rna=True, out_dim=128,
                         seed=42):
    torch.manual_seed(seed)
    letters = ['A', 'C', 'G', 'U'] if is_rna else ['A', 'C', 'G', 'T']

    seqs = [""] * len(names)
    if seq_path and os.path.exists(seq_path):
        seq_map = _load_seq_map(seq_path)
        for i, n in enumerate(names):
            if n in seq_map:
                s = str(seq_map[n]).upper().strip()
                seqs[i] = s.replace('T', 'U') if is_rna else s.replace('U', 'T')
    kmer = _kmer_features(seqs, letters, k=3)
    sec = _secondary_structure_features(seqs, is_rna=is_rna)

    sim = np.zeros((len(names), len(names)), dtype=np.float32)
    if sim_path and os.path.exists(sim_path):
        sim_row_names, sim_col_names, sim_mat = _load_sim_matrix(sim_path)
        rna_type = 'miRNA' if is_rna else 'lncRNA'
        sim = _align_sim_matrix(
            sim_row_names, sim_col_names, sim_mat, names, name=rna_type)
    else:
        np.fill_diagonal(sim, 1.0)

    kmer_t = torch.from_numpy(kmer).float()
    sec_t = torch.from_numpy(sec).float()
    sim_t = torch.from_numpy(sim).float()

    fusion = _RNASeqFusion(kmer_t.shape[1], sec_t.shape[1], sim_t.shape[1],
                           out=out_dim)
    fusion.eval()
    with torch.no_grad():
        feat = fusion(kmer_t, sec_t, sim_t)
    print(f"[Feat] {'miRNA' if is_rna else 'lncRNA'} "
          f"feat={feat.shape} (kmer={kmer_t.shape[1]}, "
          f"sec={sec_t.shape[1]}, sim={sim_t.shape[1]})")
    return feat


def extract_drug_features(names, smiles_path, out_dim=128, seed=42):
    torch.manual_seed(seed)
    n = len(names)

    if _HAS_RDKIT and smiles_path and os.path.exists(smiles_path):
        df = _read_excel(smiles_path)
        df.columns = ['Drug', 'SMILES']
        smiles_map = dict(zip(df['Drug'].str.lower().str.strip(),
                              df['SMILES']))
        morgan, maccs, daylight = [], [], []
        for name in names:
            smi = smiles_map.get(name, "")
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol:
                morgan.append(torch.tensor(
                    list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)),
                    dtype=torch.float32))
                maccs.append(torch.tensor(
                    list(MACCSkeys.GenMACCSKeys(mol)), dtype=torch.float32))
                daylight.append(torch.tensor(
                    list(rdMolDescriptors.GetMorganFingerprintAsBitVect(
                        mol, 2, nBits=2048)), dtype=torch.float32))
            else:
                morgan.append(torch.zeros(2048))
                maccs.append(torch.zeros(167))
                daylight.append(torch.zeros(2048))
        morgan_t = torch.stack(morgan)
        maccs_t = torch.stack(maccs)
        daylight_t = torch.stack(daylight)
        fusion = _DrugFusion(out=out_dim)
        fusion.eval()
        with torch.no_grad():
            feat = fusion(morgan_t, maccs_t, daylight_t)
        print(f"[Feat] drug feat={feat.shape} "
              f"(Morgan+MACCS+Daylight fusion)")
    else:
        feat = torch.eye(n, dtype=torch.float32)
        print(f"[Feat] drug feat={feat.shape} "
              f"(identity fallback; rdkit/SMILES unavailable)")
    return feat


def _sim_to_edges(sim_matrix, threshold=0.5):
    sim = np.asarray(sim_matrix, dtype=np.float32).copy()
    np.fill_diagonal(sim, 0.0)
    n = sim.shape[0]

    pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                pairs.add((i, j))

    if not pairs:
        for i in range(n):
            j = int(np.argmax(sim[i]))
            if sim[i, j] > 0 and i != j:
                pairs.add((min(i, j), max(i, j)))

    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)

    arr = np.array(sorted(pairs), dtype=np.int64).T
    return arr


def _sim_matrix_to_edges_topk(sim_matrix, top_k=5):
    sim = np.asarray(sim_matrix, dtype=np.float32).copy()
    np.fill_diagonal(sim, -1.0)
    n = sim.shape[0]
    pairs = set()
    for i in range(n):
        top = np.argsort(sim[i])[-top_k:]
        for j in top:
            if sim[i, j] > 0 and i != j:
                pairs.add((min(i, j), max(i, j)))
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(sorted(pairs), dtype=np.int64).T


def _sim_stats(sim_matrix, name=""):
    sim = np.asarray(sim_matrix, dtype=np.float32)
    off_diag = sim[~np.eye(sim.shape[0], dtype=bool)]
    return (f"[{name}] min={off_diag.min():.4f}, max={off_diag.max():.4f}, "
            f"mean={off_diag.mean():.4f}, median={np.median(off_diag):.4f}, "
            f">0.5={(off_diag >= 0.5).sum()}, >0.1={(off_diag >= 0.1).sum()}")


def _drug_similarity_edges(drug_feat, top_k=5):
    x = drug_feat.numpy()
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x_norm = x / norms
    sim = x_norm @ x_norm.T
    return _sim_matrix_to_edges_topk(sim, top_k=top_k)


def _build_offset_adjs(md_edges, ml_edges, ld_edges,
                       ll_edges, mm_edges, dd_edges, offsets, N):
    n_mirna = offsets['n_mirna']
    lnc_off = offsets['lnc_offset']
    drug_off = offsets['drug_offset']

    def _make(rows, cols, row_off=0, col_off=0):
        r = np.asarray(rows) + row_off
        c = np.asarray(cols) + col_off
        data = np.ones(len(r), dtype=np.float32)
        return sp.coo_matrix((data, (r, c)), shape=(N, N))

    adj0 = _make(md_edges[0], md_edges[1], 0, drug_off)
    adj1 = _make(md_edges[1], md_edges[0], drug_off, 0)
    adj2 = _make(ml_edges[0], ml_edges[1], 0, lnc_off)
    adj3 = _make(ml_edges[1], ml_edges[0], lnc_off, 0)
    adj4 = _make(ld_edges[0], ld_edges[1], lnc_off, drug_off)
    adj5 = _make(ld_edges[1], ld_edges[0], drug_off, lnc_off)
    adj6 = _make(ll_edges[0], ll_edges[1], lnc_off, lnc_off)
    adj7 = _make(mm_edges[0], mm_edges[1], 0, 0)
    adj8 = _make(dd_edges[0], dd_edges[1], drug_off, drug_off)

    eye = sp.eye(N, dtype=np.float32)
    adjs = [
        sparse_to_torch(normalize_row(adj0 + eye)),
        sparse_to_torch(normalize_row(adj1 + eye)),
        sparse_to_torch(normalize_row(adj2 + eye)),
        sparse_to_torch(normalize_row(adj3 + eye)),
        sparse_to_torch(normalize_row(adj4 + eye)),
        sparse_to_torch(normalize_row(adj5 + eye)),
        sparse_to_torch(normalize_sym(adj6 + eye)),
        sparse_to_torch(normalize_sym(adj7 + eye)),
        sparse_to_torch(normalize_sym(adj8 + eye)),
        sparse_to_torch(sp.eye(N, dtype=np.float32).tocoo()),
        torch.sparse_coo_tensor(size=(N, N)),
    ]
    print(f"[Data] Built {len(adjs)} adjacency matrices (N={N})")
    return adjs


def sample_negatives(adj_row, pro_pm, pro_zr, n_drug):
    B = adj_row.shape[0]
    device = adj_row.device
    k_pm = torch.zeros_like(adj_row)
    k_zr = torch.zeros_like(adj_row)
    adj_np = adj_row.cpu().numpy()
    for i in range(B):
        pos = np.where(adj_np[i] != 0)[0]
        candidates = list(set(range(n_drug)) - set(pos.tolist()))
        random.shuffle(candidates)
        if len(candidates) >= pro_pm + pro_zr:
            pm_idx = candidates[:pro_pm]
            zr_idx = candidates[pro_pm:pro_pm + pro_zr]
        else:
            half = len(candidates) // 2
            pm_idx = candidates[:half]
            zr_idx = candidates[half:]
        k_pm[i, pm_idx] = 1.0
        k_zr[i, zr_idx] = 1.0
    return k_pm.to(device), k_zr.to(device)


def sample_negative_pairs(n_mirna, n_drug, pos_edges, num_neg, device='cpu'):
    pos_set = set((int(r), int(c)) for r, c in pos_edges)
    neg = []
    while len(neg) < num_neg:
        r = random.randint(0, n_mirna - 1)
        c = random.randint(0, n_drug - 1)
        if (r, c) not in pos_set and (r, c) not in neg:
            neg.append((r, c))
    return torch.tensor(neg, dtype=torch.long, device=device)


def load_heterograph_data(data_dir, device='cpu', fold=None, n_folds=5, seed=42,
                      feat_dim=128, test_ratio=0.2, sim_top_k=5,
                      val_ratio=0.1):
    mirna_name, lnc_name, drug_name, md_edges, ld_edges, ml_edges = \
        load_associations(data_dir)

    n_mirna = len(mirna_name)
    n_lnc = len(lnc_name)
    n_drug = len(drug_name)
    N = n_mirna + n_lnc + n_drug
    offsets = {
        'n_mirna': n_mirna, 'n_lncRNA': n_lnc, 'n_drug': n_drug,
        'n_gene': n_lnc,
        'N': N,
        'gene_offset': n_mirna,
        'lnc_offset': n_mirna,
        'drug_offset': n_mirna + n_lnc,
    }

    mirna_feat = extract_rna_features(
        mirna_name,
        os.path.join(data_dir, 'miRNA_sequences.xlsx'),
        os.path.join(data_dir, 'miRNA_similarity_matrix.xlsx'),
        is_rna=True, out_dim=feat_dim, seed=seed)
    lnc_feat = extract_rna_features(
        lnc_name,
        os.path.join(data_dir, 'lncRNA_sequences.xlsx'),
        os.path.join(data_dir, 'lncRNA_similarity_matrix.xlsx'),
        is_rna=False, out_dim=feat_dim, seed=seed)
    drug_feat = extract_drug_features(
        drug_name,
        os.path.join(data_dir, 'drug_smiles.xlsx'),
        out_dim=feat_dim, seed=seed)
    node_feats = [mirna_feat, lnc_feat, drug_feat]

    node_types = torch.zeros(N, dtype=torch.long)
    node_types[n_mirna:n_mirna + n_lnc] = 1
    node_types[n_mirna + n_lnc:] = 2

    mm_edges = np.zeros((2, 0), dtype=np.int64)
    mirna_sim_path = os.path.join(data_dir, 'miRNA_similarity_matrix.xlsx')
    if os.path.exists(mirna_sim_path):
        try:
            sim_row_names, sim_col_names, sim_mat = _load_sim_matrix(mirna_sim_path)
            aligned = _align_sim_matrix(
                sim_row_names, sim_col_names, sim_mat, mirna_name, name='miRNA')
            print(f"[Data] miRNA sim matrix: {_sim_stats(aligned, 'miRNA')}")
            mm_edges = _sim_matrix_to_edges_topk(aligned, top_k=sim_top_k)
        except Exception as e:
            print(f"[Data] miRNA sim graph skipped: {e}")
    print(f"[Data] miRNA-miRNA similarity edges: {mm_edges.shape[1]} (top_k={sim_top_k})")

    ll_edges = np.zeros((2, 0), dtype=np.int64)
    lnc_sim_path = os.path.join(data_dir, 'lncRNA_similarity_matrix.xlsx')
    if os.path.exists(lnc_sim_path):
        try:
            sim_row_names, sim_col_names, sim_mat = _load_sim_matrix(lnc_sim_path)
            aligned = _align_sim_matrix(
                sim_row_names, sim_col_names, sim_mat, lnc_name, name='lncRNA')
            print(f"[Data] lncRNA sim matrix: {_sim_stats(aligned, 'lncRNA')}")
            ll_edges = _sim_matrix_to_edges_topk(aligned, top_k=sim_top_k)
        except Exception as e:
            print(f"[Data] lncRNA sim graph skipped: {e}")
    print(f"[Data] lncRNA-lncRNA similarity edges: {ll_edges.shape[1]} (top_k={sim_top_k})")

    dd_edges = _drug_similarity_edges(drug_feat, top_k=sim_top_k)
    print(f"[Data] drug-drug similarity edges: {dd_edges.shape[1]} (top_k={sim_top_k})")

    rng = np.random.RandomState(seed)
    n_md = md_edges.shape[1]

    if fold is not None and n_folds > 1:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kf.split(range(n_md)))
        trainval_idx, test_idx = splits[fold]
    else:
        perm = rng.permutation(n_md)
        n_test = int(n_md * test_ratio)
        test_idx = perm[:n_test]
        trainval_idx = perm[n_test:]

    test_pos = md_edges[:, test_idx].T
    trainval_pos = md_edges[:, trainval_idx].T

    pos_set = set((int(r), int(c)) for r, c in md_edges.T)
    test_neg = []
    while len(test_neg) < len(test_pos):
        r = rng.randint(0, n_mirna)
        c = rng.randint(0, n_drug)
        if (r, c) not in pos_set and (r, c) not in test_neg:
            test_neg.append((r, c))
    test_neg = np.array(test_neg, dtype=np.int64)
    test_edges = np.concatenate([test_pos, test_neg], axis=0)
    test_labels = np.concatenate([
        np.ones(len(test_pos), dtype=np.float32),
        np.zeros(len(test_neg), dtype=np.float32)])

    n_trainval = len(trainval_pos)
    val_seed = seed + (fold + 1) * 1000 if fold is not None else seed
    rng_val = np.random.RandomState(val_seed)
    perm2 = rng_val.permutation(n_trainval)
    n_val = max(1, int(n_trainval * val_ratio))
    val_idx = perm2[:n_val]
    train_idx = perm2[n_val:]

    train_pos = trainval_pos[train_idx]
    val_pos = trainval_pos[val_idx]

    adj_train = np.zeros((n_mirna, n_drug), dtype=np.float32)
    adj_train[train_pos[:, 0], train_pos[:, 1]] = 1.0

    md_train = train_pos.T
    adjs = _build_offset_adjs(
        md_train, ml_edges, ld_edges, ll_edges, mm_edges, dd_edges,
        offsets, N)

    adjs = [a.to(device) for a in adjs]
    node_feats = [f.to(device) for f in node_feats]
    node_types = node_types.to(device)
    adj_train = torch.from_numpy(adj_train).float().to(device)

    print(f"[Data] train_pos={len(train_pos)}, "
          f"val_pos={len(val_pos)}, test_pos={len(test_pos)}, "
          f"test_neg={len(test_neg)}")

    return {
        'adjs': adjs,
        'node_feats': node_feats,
        'node_types': node_types,
        'offsets': offsets,
        'train_pos': train_pos,
        'val_pos': val_pos,
        'test_edges': test_edges,
        'test_labels': test_labels,
        'adj_train': adj_train,
        'n_adjs': len(adjs),
    }
