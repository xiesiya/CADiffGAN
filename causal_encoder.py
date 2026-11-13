import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Op(nn.Module):
    def forward(self, x, adjs, weight, idx):
        return weight[idx] * torch.spmm(adjs[idx], x)


class CausalAttention(nn.Module):
    def __init__(self, n_hid, n_adjs, edge_emb_dim=32):
        super().__init__()
        self.n_hid = n_hid
        self.edge_embedding = nn.Embedding(n_adjs, edge_emb_dim)
        self.q_proj = nn.Linear(n_hid, n_hid)
        self.k_proj = nn.Linear(n_hid + edge_emb_dim, n_hid)
        self.v_proj = nn.Linear(n_hid, n_hid)
        self.out_proj = nn.Linear(n_hid, n_hid)
        self.scale = n_hid ** -0.5
        self.norm = nn.LayerNorm(n_hid)

    def forward(self, states, selected_edges):
        T = len(states)
        h = torch.stack(states, dim=1)
        N = h.size(0)
        device = h.device

        edge_idx = torch.tensor(selected_edges, dtype=torch.long, device=device)
        edge_embs = self.edge_embedding(edge_idx)
        edge_embs = edge_embs.unsqueeze(0).expand(N, -1, -1)

        q = self.q_proj(h)
        k = self.k_proj(torch.cat([h, edge_embs], dim=-1))
        v = self.v_proj(h)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        causal_mask = torch.triu(
            torch.ones(T, T, device=device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        context = torch.bmm(attn, v)
        context = self.out_proj(context)

        out = self.norm(h + context)

        out_last = out[:, -1, :]

        eps = 1e-10
        entropy = -(attn * torch.log(attn + eps)).sum(dim=-1)
        max_ent = torch.log(torch.tensor(float(T), device=attn.device))
        entropy_norm = entropy / max_ent
        node_scores = 1.0 - entropy_norm.mean(dim=1)

        return out_last, node_scores


class CausalMetaGraphCell(nn.Module):
    def __init__(self, n_step, n_hid, n_adjs, cstr,
                 use_norm=True, use_nl=True):
        super().__init__()
        self.n_step = n_step
        self.n_adjs = n_adjs
        self.cstr = cstr

        self.affine = nn.Linear(n_hid, n_hid)
        self.norm = (nn.LayerNorm(n_hid, elementwise_affine=False)
                     if use_norm else lambda x: x)
        self.use_nl = use_nl

        self.ops_seq = nn.ModuleList([Op() for _ in range(1, n_step)])
        n_res = sum(i - 1 for i in range(2, n_step))
        self.ops_res = nn.ModuleList([Op() for _ in range(n_res)])
        self.last_seq = Op()
        self.last_res = nn.ModuleList([Op() for _ in range(n_step - 1)])

    def forward(self, x, adjs, ws_seq, idxes_seq, ws_res, idxes_res):
        x = self.affine(x)
        states = [x]
        selected_edges = []
        offset = 0

        for i in range(self.n_step - 1):
            edge_idx = idxes_seq[0][i]
            seqi = self.ops_seq[i](
                states[i], adjs[:-1], ws_seq[0][i], edge_idx)
            selected_edges.append(
                edge_idx.item() if torch.is_tensor(edge_idx) else edge_idx)

            resi = sum(
                self.ops_res[offset + j](
                    h, adjs, ws_res[0][offset + j], idxes_res[0][offset + j])
                for j, h in enumerate(states[:i]))
            offset += i
            states.append(seqi + resi)

        adjs_cstr = [adjs[i] for i in self.cstr]
        last_edge_idx = idxes_seq[1]
        out_seq = self.last_seq(
            states[-1], adjs_cstr, ws_seq[1], last_edge_idx)
        selected_edges.append(
            self.cstr[last_edge_idx.item()]
            if torch.is_tensor(last_edge_idx)
            else self.cstr[last_edge_idx])

        adjs_cstr.append(adjs[-1])
        out_res = sum(
            self.last_res[i](
                h, adjs_cstr, ws_res[1][i], idxes_res[1][i])
            for i, h in enumerate(states[:-1]))

        output = self.norm(out_seq + out_res)
        if self.use_nl:
            output = F.gelu(output)
        states.append(output)

        return output, states, selected_edges


class CausalMetaGraphEncoder(nn.Module):
    def __init__(self, in_dims, n_hid, n_adjs, n_steps, cstr,
                 attn_dim=64, edge_emb_dim=32, dropout=0.2,
                 use_norm=True, out_nl=True):
        super().__init__()
        self.n_hid = n_hid
        self.n_adjs = n_adjs
        self.cstr = cstr
        self.n_steps = n_steps

        self.ws = nn.ModuleList([
            nn.Linear(in_dims[i], n_hid) for i in range(len(in_dims))])

        self.metas = nn.ModuleList([
            CausalMetaGraphCell(
                n_steps[i], n_hid, n_adjs, cstr,
                use_norm=use_norm, use_nl=out_nl)
            for i in range(len(n_steps))])

        self.causal_attns = nn.ModuleList([
            CausalAttention(n_hid, n_adjs, edge_emb_dim)
            for _ in range(len(n_steps))])

        self.attn_fc1 = nn.Linear(n_hid, attn_dim)
        self.attn_fc2 = nn.Linear(attn_dim, 1)
        self.feats_drop = nn.Dropout(dropout)

        self.as_seq = []
        self.as_last_seq = []
        for i in range(len(n_steps)):
            if n_steps[i] > 1:
                ai = 1e-3 * torch.randn(n_steps[i] - 1, n_adjs - 1)
                ai = nn.Parameter(ai)
                self.register_parameter(f'as_seq_{i}', ai)
                self.as_seq.append(ai)
            else:
                self.as_seq.append(None)

            ai_last = 1e-3 * torch.randn(len(cstr))
            ai_last = nn.Parameter(ai_last)
            self.register_parameter(f'as_last_seq_{i}', ai_last)
            self.as_last_seq.append(ai_last)

        ks = [sum(1 for i in range(2, n_steps[k]) for _ in range(i - 1))
              for k in range(len(n_steps))]
        self.as_res = []
        self.as_last_res = []
        for i in range(len(n_steps)):
            if ks[i] > 0:
                ai = 1e-3 * torch.randn(ks[i], n_adjs)
                ai = nn.Parameter(ai)
                self.register_parameter(f'as_res_{i}', ai)
                self.as_res.append(ai)
            else:
                self.as_res.append(None)

            if n_steps[i] > 1:
                ai_last = 1e-3 * torch.randn(n_steps[i] - 1, len(cstr) + 1)
                ai_last = nn.Parameter(ai_last)
                self.register_parameter(f'as_last_res_{i}', ai_last)
                self.as_last_res.append(ai_last)
            else:
                self.as_last_res.append(None)

    def alphas(self):
        result = []
        for a in self.as_seq:
            if a is not None:
                result.append(a)
        for a in self.as_last_seq:
            result.append(a)
        for a in self.as_res:
            if a is not None:
                result.append(a)
        for a in self.as_last_res:
            if a is not None:
                result.append(a)
        return result

    def sample(self, eps):
        idxes_seq = []
        idxes_res = []
        if np.random.uniform() < eps:
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_seq[i] is None else
                            torch.randint(0, self.as_seq[i].size(-1),
                                          self.as_seq[i].size()[:-1]))
                temp.append(torch.randint(0, self.as_last_seq[i].size(-1), (1,)))
                idxes_seq.append(temp)
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_res[i] is None else
                            torch.randint(0, self.as_res[i].size(-1),
                                          self.as_res[i].size()[:-1]))
                temp.append(None if self.as_last_res[i] is None else
                            torch.randint(0, self.as_last_res[i].size(-1),
                                          self.as_last_res[i].size()[:-1]))
                idxes_res.append(temp)
        else:
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_seq[i] is None else
                            torch.argmax(F.softmax(self.as_seq[i], dim=-1), dim=-1))
                temp.append(torch.argmax(F.softmax(self.as_last_seq[i], dim=-1), dim=-1))
                idxes_seq.append(temp)
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_res[i] is None else
                            torch.argmax(F.softmax(self.as_res[i], dim=-1), dim=-1))
                temp.append(None if self.as_last_res[i] is None else
                            torch.argmax(F.softmax(self.as_last_res[i], dim=-1), dim=-1))
                idxes_res.append(temp)
        return idxes_seq, idxes_res

    def forward(self, node_feats, node_types, adjs, idxes_seq, idxes_res):
        device = node_feats[0].device
        N = node_types.size(0)

        hid = torch.zeros(N, self.n_hid, device=device)
        for i in range(len(node_feats)):
            mask = node_types == i
            hid[mask] = self.ws[i](node_feats[i])
        hid = self.feats_drop(hid)

        cell_outputs = []
        cell_attns = []
        for i, meta in enumerate(self.metas):
            ws_seq = [
                None if self.as_seq[i] is None
                else F.softmax(self.as_seq[i], dim=-1),
                F.softmax(self.as_last_seq[i], dim=-1)]
            ws_res = [
                None if self.as_res[i] is None
                else F.softmax(self.as_res[i], dim=-1),
                None if self.as_last_res[i] is None
                else F.softmax(self.as_last_res[i], dim=-1)]

            cell_out, states, sel_edges = meta(
                hid, adjs, ws_seq, idxes_seq[i], ws_res, idxes_res[i])

            caus_out, node_scores = self.causal_attns[i](
                states[1:], sel_edges)
            cell_outputs.append(caus_out)
            cell_attns.append(node_scores)

        hids = torch.stack(cell_outputs, dim=0).transpose(0, 1)
        attns = []
        for c_out in cell_outputs:
            a = self.attn_fc2(torch.tanh(self.attn_fc1(c_out)))
            attns.append(a)
        attns = F.softmax(torch.cat(attns, dim=-1), dim=-1)
        node_emb = (attns.unsqueeze(-1) * hids).sum(dim=1)

        node_attn = torch.stack(cell_attns, dim=0).mean(dim=0)

        return node_emb, node_attn

    def parse(self):
        idxes_seq, idxes_res = self.sample(0.0)
        msg_seq = []
        for i in range(len(idxes_seq)):
            map_seq = [self.cstr[idxes_seq[i][1].item()]]
            msg_seq.append(
                map_seq if idxes_seq[i][0] is None
                else idxes_seq[i][0].tolist() + map_seq)
        return msg_seq
