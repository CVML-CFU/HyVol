import torch
import torch.nn as nn
import torch.nn.functional as F

# Batch hypergraph over the NON-TEXT modalities only. Text is never a vertex (leak-free): a refined
# gallery embedding must not absorb its own caption. Doc hyperedges are block-diagonal; semantic
# mutual-kNN edges are the only cross-doc wiring and exist at train time only.


def doc_incidence(B, mask, device, present=None):
    """H_doc (|V|, B): hyperedge j connects the non-text vertices of doc j. Block-diagonal.

    present (B, k1) 0/1: a MISSING modality (zero-filled feature) is disconnected from its doc edge
    (weight 0), so it neither feeds nor receives graph messages and never pollutes the fusion. None
    => every vertex connected (complete-modality behaviour, unchanged)."""
    k1 = len(mask)
    H = torch.zeros(B * k1, B, device=device)
    idx = torch.arange(B, device=device)
    for m in range(k1):
        H[idx * k1 + m, idx] = 1.0 if present is None else present[:, m]
    return H


@torch.no_grad()
def mutual_knn_adj(t_frozen, k=4, edge_dropout=0.3, training=True, sim_std=None):

    B = t_frozen.shape[0]
    k = min(k, max(2, B // 4))                                 # adaptive: selective fraction of the shard
    if B <= k + 1:
        return None
    t = F.normalize(t_frozen.float(), dim=-1)
    sim = t @ t.T
    sim.fill_diagonal_(float('-inf'))
    nn_idx = sim.topk(k, dim=-1).indices
    sel = torch.zeros(B, B, device=t.device)
    sel.scatter_(1, nn_idx, 1.0)
    mutual = sel * sel.T                                       # binary mutual mask = edge STRUCTURE (top-k mutual)
    adj = mutual                                               # STRUCTURE only; edge WEIGHTS are LEARNED in GatedHGNN.edge_attention (leak-free)
    if sim_std is not None and sim_std > 0:
        off = sim[sim > -1e30]
        thr = off.mean() + sim_std * off.std()                # adaptive per-batch floor
        adj = adj * (sim >= thr).float()
    if training and edge_dropout > 0:
        keep = (torch.rand(B, B, device=t.device) > edge_dropout).float()
        keep = torch.minimum(keep, keep.T)                    # keep symmetric
        adj = adj * keep
    return adj


def semantic_incidence(adj, B, mask, device):
    """H_sem (|V|, B): semantic edge j connects the non-text vertices of doc j and its mutual
    neighbours. adj: (B, B) from mutual_knn_adj (or None)."""
    if adj is None or adj.sum() == 0:
        return None
    k1 = len(mask)
    members = adj + torch.eye(B, device=device)
    return members.repeat_interleave(k1, dim=0)


class GatedHGNN(nn.Module):
    """<=2 layers of V->E->V message passing with a zero-init gated residual: at step 0 the model
    is the plain AL-heads baseline; the graph only refines.

        F_E  = GELU( D_E^-1 H^T  F_V W_V )
        F_Vn = GELU( D_V^-1 H    F_E W_E )
        F_V <- F_V + tanh(gate_l) * F_Vn
    """

    def __init__(self, k=512, n_layers=2):
        super().__init__()
        assert n_layers <= 2, 'more layers = over-smoothing'
        self.n_layers = n_layers
        self.W_V = nn.ModuleList([nn.Linear(k, k, bias=False) for _ in range(n_layers)])
        self.W_E = nn.ModuleList([nn.Linear(k, k, bias=False) for _ in range(n_layers)])
        self.gates = nn.Parameter(torch.full((n_layers,), 1.0))
        self.edge_head = nn.Linear(k, k, bias=False)
        # LEARNED attention for semantic (cross-doc) edge weights — GAT-style, leak-free (runs on
        # detached text). small init => near-uniform start, and BOTH attn_w & attn_a learn from step 0.
        self.attn_w = nn.Linear(k, k, bias=False)
        self.attn_a = nn.Parameter(torch.randn(2 * k) * 0.1)

    def edge_attention(self, t_frozen, mask_bin):
        """Learned GAT-style attention weights on the mutual edges. t_frozen is detached upstream
        (leak-free: the text encoder gets NO gradient from the graph; only attn_w/attn_a learn).
        mask_bin (B,B) 0/1 = which edges exist. Returns a symmetric weighted adjacency (B,B)."""
        t = F.normalize(t_frozen.float(), dim=-1)
        h = self.attn_w(t)                                        # (B, K)
        K = h.shape[1]
        e = (h @ self.attn_a[:K]).unsqueeze(1) + (h @ self.attn_a[K:]).unsqueeze(0)   # additive attn (B,B)
        e = F.leaky_relu(e, 0.2)
        e = e.masked_fill(mask_bin <= 0, torch.finfo(e.dtype).min)   # only over real edges
        attn = torch.softmax(e, dim=1)                            # normalize over each node's neighbours
        has_edge = (mask_bin > 0).any(dim=1, keepdim=True)        # guard rows with no edge (avoid NaN)
        attn = torch.where(has_edge, attn, torch.zeros_like(attn)) * (mask_bin > 0).float()
        return 0.5 * (attn + attn.T)                              # symmetric weighted adjacency

    @staticmethod
    def _norm(H):
        d = H.sum(dim=0).clamp(min=1.0)                        # edge degree
        dv = H.sum(dim=1).clamp(min=1.0)                       # vertex degree
        return H / d.unsqueeze(0), H / dv.unsqueeze(1)

    def forward(self, z, mask, H_doc, H_sem=None, present=None):
        """z: dict non-text modality -> (B, K) L2-normed (no 'T'). present (B,k1) 0/1 marks which
        modalities a doc actually has; a missing one is masked out of the doc edge (H_doc) and of the
        fusion mean, so it neither passes messages nor dilutes h. present=None -> complete behaviour.
        Returns z_hat (L2-normed non-text mods), h (B,K) L2-normed doc embedding, h_prenorm."""
        order = tuple(mask)
        B = z[order[0]].shape[0]
        k1 = len(order)
        F_V = torch.stack([z[m] for m in order], dim=1).reshape(B * k1, -1)

        H = H_doc if H_sem is None else torch.cat([H_doc, H_sem], dim=1)
        H_e, H_v = self._norm(H)

        for l in range(self.n_layers):
            F_E = F.gelu(H_e.T @ self.W_V[l](F_V))
            _msg = H_v @ self.W_E[l](F_E)
            F_Vn = F.gelu(_msg) if l < self.n_layers - 1 else _msg
            F_V = F_V + torch.tanh(self.gates[l]) * F_Vn

        V = F_V.reshape(B, k1, -1)
        z_hat = {m: F.normalize(V[:, i], dim=-1) for i, m in enumerate(order)}
        if present is None:
            h_prenorm = self.edge_head(V.mean(dim=1))
        else:
            w = present.unsqueeze(-1)                                      # (B, k1, 1)
            h_prenorm = self.edge_head((V * w).sum(1) / w.sum(1).clamp(min=1.0))   # mean over PRESENT only
        h = F.normalize(h_prenorm, dim=-1)
        return z_hat, h, h_prenorm
