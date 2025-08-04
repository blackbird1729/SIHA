import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(num_timesteps * input_dim, hidden_dim)
        # self.input_proj = nn.Sequential(
        #     nn.Linear(num_timesteps * input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU()
        # )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, adj=None):

        B, V, T, D = x.size()
        x = x.view(B, V, T * D)
        h = self.input_proj(x)  # [B, V, H]

        # 构造邻接矩阵：全连接或使用GT结构
        if adj is None:
            adj = torch.ones(B, V, V, device=x.device) - torch.eye(self.V, device=x.device).unsqueeze(0)
        else:
            adj=adj[:,:V,:V]
            assert adj.shape == (B, V, V)

        # 扩展所有节点输入


        h_i = h.unsqueeze(2).expand(-1, -1, V, -1)
        h_j = h.unsqueeze(1).expand(-1, V, -1, -1)
        edge_input = torch.cat([h_i, h_j], dim=-1)  # [B, N, N, 2H]
        edge_feat = self.edge_mlp(edge_input)       # [B, N, N, H]
        edge_feat = edge_feat * adj.unsqueeze(-1)

        agg = edge_feat.sum(dim=2)  # [B, N, H]
        node_feat = self.node_mlp(agg)  # [B, N, H]

        out = self.out_proj(node_feat)  # [B, N, H]
        return out

class HiddenStatePredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps, num_hidden_nodes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_hidden_nodes * num_timesteps * output_dim)
        )
        self.num_hidden = num_hidden_nodes
        self.T = num_timesteps
        self.D = output_dim

    def forward(self, x):  # x: [B, V, T, D]
        B, V, T, D = x.size()
        x = x.view(B, -1)
        out = self.fc(x)
        return out.view(B, self.num_hidden, self.T, self.D)


class HiddenStatePredictorTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps, num_hidden_nodes, num_layers=2, nhead=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim,batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, num_hidden_nodes * num_timesteps * output_dim)
        self.T = num_timesteps
        self.D = output_dim
        self.num_hidden = num_hidden_nodes

    def forward(self, x):  # x: [B, V, T, D]
        B, V, T, D = x.size()
        x = x.view(B, V * T, D)  # [B, V*T, D]
        x = self.input_proj(x)  # [B, V*T, H]
        x = x.permute(1, 0, 2)  # [S, B, H]
        x = self.transformer(x)  # [S, B, H]
        x = x.mean(dim=0)  # [B, H]
        x = self.out_proj(x)  # [B, H * D]
        return x.view(B, self.num_hidden, self.T, self.D)

class GNNHiddenStatePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps, num_hidden_nodes, num_visible_nodes):
        super().__init__()
        self.T = num_timesteps
        self.D = output_dim
        self.H = num_hidden_nodes
        self.V = num_visible_nodes
        self.N = self.V + self.H
        self.hidden_dim = hidden_dim

        # self.input_proj = nn.Linear(num_timesteps * input_dim, hidden_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(num_timesteps * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.out_proj = nn.Linear(hidden_dim, num_timesteps * output_dim)

    def forward(self, x, adj=None):
        B, V, T, D = x.size()
        x = x.view(B, V, T * D)
        h = self.input_proj(x)  # [B, V, H]

        # 构造邻接矩阵：全连接或使用GT结构
        if adj is None:
            adj = torch.ones(B, self.N, self.N, device=x.device) - torch.eye(self.N, device=x.device).unsqueeze(0)
        else:
            assert adj.shape == (B, self.N, self.N)

        # 扩展所有节点输入
        h_full = torch.zeros(B, self.N, self.hidden_dim, device=x.device)
        h_full[:, :V] = h  # 前 V 个是可见节点

        h_i = h_full.unsqueeze(2).expand(-1, -1, self.N, -1)
        h_j = h_full.unsqueeze(1).expand(-1, self.N, -1, -1)
        edge_input = torch.cat([h_i, h_j], dim=-1)  # [B, N, N, 2H]
        edge_feat = self.edge_mlp(edge_input)       # [B, N, N, H]
        edge_feat = edge_feat * adj.unsqueeze(-1)

        agg = edge_feat.sum(dim=2)  # [B, N, H]
        node_feat = self.node_mlp(agg)  # [B, N, H]

        hidden_pred = self.out_proj(node_feat[:, self.V:])  # [B, H, T*D]
        return hidden_pred.view(B, self.H, self.T, self.D)


class GATHiddenStatePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps, num_hidden_nodes, num_visible_nodes):
        super().__init__()
        self.T = num_timesteps
        self.D = output_dim
        self.H = num_hidden_nodes
        self.V = num_visible_nodes
        self.N = self.V + self.H
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(num_timesteps * input_dim, hidden_dim)

        # attention parameter: [2H → 1]
        self.attn_proj = nn.Linear(2 * hidden_dim, 1)

        # feature update MLP
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.out_proj = nn.Linear(hidden_dim, num_timesteps * output_dim)

    def forward(self, x, adj=None):
        B, V, T, D = x.shape
        x = x.view(B, V, T * D)                    # [B, V, TD]
        h_vis = self.input_proj(x)                 # [B, V, H]

        # 初始化所有节点：可见+隐藏
        h_full = torch.zeros(B, self.N, self.hidden_dim, device=x.device)
        h_full[:, :V] = h_vis                      # [B, N, H]

        # 构造邻接矩阵（默认全连接无自环）
        if adj is None:
            adj = torch.ones(B, self.N, self.N, device=x.device) - torch.eye(self.N, device=x.device).unsqueeze(0)
        else:
            assert adj.shape == (B, self.N, self.N)

        # 构造所有边 h_i || h_j
        h_i = h_full.unsqueeze(2).expand(-1, -1, self.N, -1)  # [B, N, N, H]
        h_j = h_full.unsqueeze(1).expand(-1, self.N, -1, -1)
        edge_cat = torch.cat([h_i, h_j], dim=-1)              # [B, N, N, 2H]

        # 计算注意力权重
        attn_logits = self.attn_proj(edge_cat).squeeze(-1)    # [B, N, N]

        attn_weight = F.softmax(attn_logits, dim=-1)          # [B, N, N]
        attn_weight = attn_weight.masked_fill(adj == 0, 0)

        # 消息：apply msg_mlp + attention
        msg = self.msg_mlp(h_j)                               # [B, N, N, H]
        msg_weighted = attn_weight.unsqueeze(-1) * msg        # [B, N, N, H]

        agg = msg_weighted.sum(dim=2)                         # [B, N, H]
        node_feat = self.node_mlp(agg)                        # [B, N, H]

        # 输出预测隐藏节点
        hidden_feat = node_feat[:, self.V:]                   # [B, H, H]
        out = self.out_proj(hidden_feat)                      # [B, H, T*D]
        return out.view(B, self.H, self.T, self.D)


class GNN_RNN_HiddenStatePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps, num_hidden_nodes, num_visible_nodes):
        super().__init__()
        self.T = num_timesteps
        self.D = output_dim
        self.H = num_hidden_nodes
        self.V = num_visible_nodes
        self.N = self.V + self.H
        self.hidden_dim = hidden_dim

        # 时间步嵌入
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 消息传递 MLP（可替换为 GAT）
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 对每个隐藏节点进行时间建模的 GRU
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # 输出预测头
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adj=None):
        B, V, T, D = x.shape
        device = x.device

        # 初始化结构（GT or fully-connected）
        if adj is None:
            adj = torch.ones(B, self.V, self.V, device=device) - torch.eye(self.V, device=device).unsqueeze(0)

        # 初始化输出
        hidden_states = []

        # 初始化 RNN 的初始状态
        h0 = torch.zeros(1, B * self.H, self.hidden_dim, device=device)

        # 初始化隐藏节点特征（可学习 or 零）
        hidden_features = torch.zeros(B, self.H, self.hidden_dim, device=device)

        for t in range(T):
            xt = x[:, :, t, :]  # [B, V, D]
            h = self.input_proj(xt)  # [B, V, H]

            # GNN 消息传递
            hi = h.unsqueeze(2).expand(-1, -1, self.V, -1)
            hj = h.unsqueeze(1).expand(-1, self.V, -1, -1)
            edge_input = torch.cat([hi, hj], dim=-1)
            edge_feat = self.edge_mlp(edge_input)
            edge_feat = edge_feat * adj.unsqueeze(-1)
            agg = edge_feat.sum(dim=2)
            node_feat = self.node_mlp(agg)  # [B, V, H]

            # 将可见节点信息传入隐藏节点 (可更复杂地通过attention)
            hidden_input = node_feat.mean(dim=1, keepdim=True).expand(-1, self.H, -1)
            hidden_features = hidden_features + hidden_input  # 融合信息

            hidden_states.append(hidden_features.unsqueeze(2))  # [B, H, 1, H]

        # [B, H, T, H] → [B*H, T, H]
        rnn_input = torch.cat(hidden_states, dim=2).permute(0, 1, 2, 3).reshape(B * self.H, T, self.hidden_dim)

        # GRU 处理
        rnn_output, _ = self.rnn(rnn_input, h0)  # [B*H, T, H]

        # 输出层
        out = self.out_proj(rnn_output)  # [B*H, T, D]
        out = out.view(B, self.H, T, self.D)
        return out


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        hidden_dim = max(dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.input_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
    def forward(self, inputs, num_slots=None):
        b, n, t, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
        slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)
        # inputs = inputs.view(b,n,-1)

        inputs = self.input_proj(inputs.view(b, n, -1))

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale

            attn = dots.softmax(dim=-1)  # softmax over inputs (tokens), shape [B, S, N]
            attn = attn / attn.sum(dim=1, keepdim=True)  # Normalize across slots

            # # attn = dots.softmax(dim=1) + self.eps
            # attn = torch.sigmoid(dots)  # or relu(dots) + eps, or (tanh(dots)+1)/2
            # attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', v, attn)
            slots = self.gru(
                updates.reshape(-1, d*t),
                slots_prev.reshape(-1, d*t)
            )
            slots = slots.reshape(b, n, -1)
            slots = slots + self.mlp(self.norm_pre_ff(slots))


        return self.out_proj(slots).view(b, self.num_slots, t, d)

        # return slots.view(b, self.num_slots, t, d)

class SetTransformerHSP(nn.Module):
    def __init__(self, input_dim, num_timesteps, num_slots, output_dim,
                 num_inds=32, dim_hidden=128, num_heads=4, ln=False,dropout=0.1):
        super(SetTransformerHSP, self).__init__()
        self.num_slots = num_slots
        self.num_timesteps = num_timesteps
        self.output_dim = output_dim
        self.gnnenc=GNN(4, dim_hidden, output_dim, num_timesteps)
        self.sabenc = nn.Sequential(
            SAB(input_dim, dim_hidden, num_heads,ln=ln, dropout=dropout),
            SAB(dim_hidden, dim_hidden, num_heads,ln=ln,dropout=dropout),
        )
        # self.enc = nn.Sequential(
        #     ISAB(input_dim, dim_hidden, num_heads, num_inds, ln=ln),
        #     ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        # )
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_slots, ln=ln, dropout=dropout),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln, dropout=dropout),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln, dropout=dropout),
            nn.Linear(dim_hidden, num_timesteps * output_dim)
        )

    def forward(self, x,adj=None):
        # x: [B, V, T, D]
        B, V, T, D = x.shape

        if adj is None:
            x = x.view(B, V, T * D)  # [B, V, T*D]
            out = self.dec(self.sabenc(x))  # [B, H, T*D]
        else:
            out = self.dec(self.gnnenc(x,adj))
        return out.view(B, self.num_slots, self.num_timesteps, self.output_dim)

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False,dropout=0.1):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.dropout = nn.Dropout(dropout)
    def forward(self, Q, K,structure_bias=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        if structure_bias is not None:
            structure_bias = structure_bias.repeat(self.num_heads, 1, 1)  # 适配多头注意力
            A = A + structure_bias
        # print(A.shape)
        A = torch.softmax(A, 2)
        A = self.dropout(A)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + self.dropout(F.relu(self.fc_o(O)))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False,dropout=0.1):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln, dropout=dropout)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False,dropout=0.1):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln, dropout=dropout)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SetTransformerHSP_sb(nn.Module):
    def __init__(self, input_dim, num_timesteps, num_slots, output_dim,
                 num_inds=32, dim_hidden=128, num_heads=4, ln=False,dropout=0.1):
        super(SetTransformerHSP_sb, self).__init__()
        self.num_slots = num_slots
        self.num_timesteps = num_timesteps
        self.output_dim = output_dim
        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.ln = ln

        self.enc = nn.Sequential(
            SAB_sb(input_dim, dim_hidden, num_heads, ln=ln,dropout=dropout),
            SAB_sb(dim_hidden, dim_hidden, num_heads, ln=ln,dropout=dropout),
        )
        self.dec = nn.Sequential(
            PMA_sb(dim_hidden, num_heads, num_slots, ln=ln,dropout=dropout),
            SAB_sb(dim_hidden, dim_hidden, num_heads, ln=ln,dropout=dropout),
            SAB_sb(dim_hidden, dim_hidden, num_heads, ln=ln,dropout=dropout),
            nn.Linear(dim_hidden, num_timesteps * output_dim)
        )

    def forward(self, x, adj=None):
        # x: [B, V, T, D], adj: [B, N, N] where N = V + H
        B, V, T, D = x.shape
        H = self.num_slots

        structure_bias_enc = None
        structure_bias_pma = None
        structure_bias_dec = None

        # if adj is not None:
        #     alpha = 0  #  Soft bias 强度（可调）
        #     structure_bias_enc = alpha * (1 - adj[:, :V, :V])     #  Encoder attention bias: 可见-可见
        #     structure_bias_pma = alpha * (1 - adj[:, V:, :V])     #  PMA decoder: 隐藏-seed <- 可见
        #     structure_bias_dec = alpha * (1 - adj[:, V:, V:])     #  Decoder SAB: 隐藏-隐藏
        if adj is not None:
            # *** 每个 attention head 使用不同的 alpha
            alphas = [-0., -1., -5., -1e9]
            # alphas = [-0., -3.0, -5.0, -1e9]  # 可调，每个 head 一项
            assert len(alphas) == self.num_heads, "alphas 数量应与 attention heads 数量一致"

            # *** 分别构造 encoder、pma、decoder 的结构 bias，得到 list of [B, Nq, Nk]
            structure_bias_enc = [alpha * (1 - adj[:, :V, :V]) for alpha in alphas]  # [H * [B, V, V]]
            structure_bias_pma = [alpha * (1 - adj[:, V:, :V]) for alpha in alphas]  # [H * [B, H, V]]
            structure_bias_dec = [alpha * (1 - adj[:, V:, V:]) for alpha in alphas]  # [H * [B, H, H]]

        x = x.view(B, V, T * D)  # [B, V, T*D]
        x_enc = self.enc[0](x, structure_bias_enc)  #  修改：传入结构 bias
        x_enc = self.enc[1](x_enc, structure_bias_enc)

        x_dec = self.dec[0](x_enc, structure_bias_pma)  # PMA
        x_dec = self.dec[1](x_dec, structure_bias_dec)
        x_dec = self.dec[2](x_dec, structure_bias_dec)
        out = self.dec[3](x_dec)

        return out.view(B, self.num_slots, self.num_timesteps, self.output_dim)


class MAB_sb(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False,dropout=0.1):
        super(MAB_sb, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.dropout = nn.Dropout(dropout)
    def forward(self, Q, K, structure_bias=None):  #  添加 structure_bias 参数
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        if structure_bias is not None:
            # *** 支持每个 head 一个结构 bias
            assert isinstance(structure_bias, list) and len(structure_bias) == self.num_heads, \
                "Expected list of bias matrices equal to num_heads"

            # *** 拼接每个 head 的结构偏置：[B, Nq, Nk] → [B*H, Nq, Nk]
            structure_bias_expanded = torch.cat([b for b in structure_bias], dim=0)
            A = A + structure_bias_expanded  # *** 每个 head 使用独立结构偏置
        # if structure_bias is not None:
        #     structure_bias = structure_bias.repeat(self.num_heads, 1, 1)  #  多头注意力结构偏置扩展
        #     A = A + structure_bias

        A = torch.softmax(A, dim=2)
        A = self.dropout(A)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + self.dropout(F.relu(self.fc_o(O)))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB_sb(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False,dropout=0.1):
        super(SAB_sb, self).__init__()
        self.mab = MAB_sb(dim_in, dim_in, dim_out, num_heads, ln=ln,dropout=dropout)

    def forward(self, X, structure_bias=None):  #  添加结构 bias 支持
        return self.mab(X, X, structure_bias)

class PMA_sb(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False,dropout=0.1):
        super(PMA_sb, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB_sb(dim, dim, dim, num_heads, ln=ln,dropout=dropout)

    def forward(self, X, structure_bias=None):  # 添加结构 bias 支持
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, structure_bias)


class PMA_sb_pretrained_seeds(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False,dropout=0.1):
        super(PMA_sb_pretrained_seeds, self).__init__()
        self.num_seeds = num_seeds
        self.inputS=nn.Linear(49*4, dim)
        self.mab = MAB_sb(dim, dim, dim, num_heads, ln=ln,dropout=dropout)

    def forward(self, X, structure_bias=None,seed_input=None):  # 添加结构 bias 支持
        seed_input = seed_input.view(seed_input.size(0), self.num_seeds ,49*4)
        S=self.inputS(seed_input)
        return self.mab(S, X, structure_bias)

class SetTransformerHSP_sb_seed(nn.Module):
    def __init__(self, input_dim, num_timesteps, num_slots, output_dim,
                 num_inds=32, dim_hidden=128, num_heads=4, ln=False,dropout=0.1):
        super(SetTransformerHSP_sb_seed, self).__init__()
        self.num_slots = num_slots
        self.num_timesteps = num_timesteps
        self.output_dim = output_dim
        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.ln = ln

        self.enc = nn.Sequential(
            SAB_sb(input_dim, dim_hidden, num_heads, ln=ln,dropout=dropout),
            SAB_sb(dim_hidden, dim_hidden, num_heads, ln=ln,dropout=dropout),
        )
        self.dec = nn.Sequential(
            PMA_sb_pretrained_seeds(dim_hidden, num_heads, num_slots, ln=ln,dropout=dropout),
            SAB_sb(dim_hidden, dim_hidden, num_heads, ln=ln,dropout=dropout),
            SAB_sb(dim_hidden, dim_hidden, num_heads, ln=ln,dropout=dropout),
            nn.Linear(dim_hidden, num_timesteps * output_dim)
        )

    def forward(self, x, adj=None, seed_input=None):
        # x: [B, V, T, D], adj: [B, N, N] where N = V + H
        B, V, T, D = x.shape
        H = self.num_slots

        structure_bias_enc = None
        structure_bias_pma = None
        structure_bias_dec = None

        # if adj is not None:
        #     alpha = 0  #  Soft bias 强度（可调）
        #     structure_bias_enc = alpha * (1 - adj[:, :V, :V])     #  Encoder attention bias: 可见-可见
        #     structure_bias_pma = alpha * (1 - adj[:, V:, :V])     #  PMA decoder: 隐藏-seed <- 可见
        #     structure_bias_dec = alpha * (1 - adj[:, V:, V:])     #  Decoder SAB: 隐藏-隐藏
        if adj is not None:
            # *** 每个 attention head 使用不同的 alpha
            alphas = [-0., -1., -5., -1e9]
            # alphas = [-0., -3.0, -5.0, -1e9]  # 可调，每个 head 一项
            assert len(alphas) == self.num_heads, "alphas 数量应与 attention heads 数量一致"

            # *** 分别构造 encoder、pma、decoder 的结构 bias，得到 list of [B, Nq, Nk]
            structure_bias_enc = [alpha * (1 - adj[:, :V, :V]) for alpha in alphas]  # [H * [B, V, V]]
            structure_bias_pma = [alpha * (1 - adj[:, V:, :V]) for alpha in alphas]  # [H * [B, H, V]]
            structure_bias_dec = [alpha * (1 - adj[:, V:, V:]) for alpha in alphas]  # [H * [B, H, H]]

        x = x.view(B, V, T * D)  # [B, V, T*D]
        x_enc = self.enc[0](x, structure_bias_enc)  #  修改：传入结构 bias
        x_enc = self.enc[1](x_enc, structure_bias_enc)

        x_dec = self.dec[0](x_enc, structure_bias_pma,seed_input)  # PMA
        x_dec = self.dec[1](x_dec, structure_bias_dec)
        x_dec = self.dec[2](x_dec, structure_bias_dec)
        out = self.dec[3](x_dec)

        return out.view(B, self.num_slots, self.num_timesteps, self.output_dim)