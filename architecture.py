import torch
import torch.nn as nn

def drm(y_true, is_treat, y_pred, with_th=True):
    msk_1 = is_treat.bool()
    msk_0 = (1 - is_treat).bool()
    y_true_1 = torch.masked_select(y_true, msk_1)
    y_true_0 = torch.masked_select(y_true, msk_0)
    y_pred_1 = torch.masked_select(y_pred, msk_1)
    y_pred_0 = torch.masked_select(y_pred, msk_0)

    if with_th:
        y_1_score = torch.softmax(torch.tanh(y_pred_1), dim=0)
        y_0_score = torch.softmax(torch.tanh(y_pred_0), dim=0)
    else:
        y_1_score = torch.softmax(y_pred_1, dim=0)
        y_0_score = torch.softmax(y_pred_0, dim=0)

    return - torch.sum(y_1_score * y_true_1) + torch.sum(y_0_score * y_true_0)


class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.bn2(out)
        return out


class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.bn2(out)
        return out


class EFIN(nn.Module):
    """
    EFIN class -- a explicit feature interaction network with two heads.
    """

    def __init__(self, x_dim,
                 sparse_x_len,
                 emb_dim,
                 num_experts,
                 experts_out,
                 experts_hidden,
                 towers_hidden,
                 tasks=2,
                 use_bn=True,
                 enable_drm=False,
                 base_coef=0.1,
                 drm_coef=10):
        super(EFIN, self).__init__()
        self.sparse_x_len = sparse_x_len
        self.enable_drm = enable_drm
        self.base_coef = base_coef
        self.drm_coef = drm_coef

        self.att_embed_1 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.att_embed_2 = nn.Linear(emb_dim, emb_dim)
        self.att_embed_3 = nn.Linear(emb_dim, 1, bias=False)


        # self-attention
        self.softmax = nn.Softmax(dim=-1)

        self.Q_w = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=True)
        self.K_w = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=True)
        self.V_w = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=True)

        # representation parts for X
        self.emb_resp = nn.Embedding(x_dim, emb_dim)
        # representation parts for T
        self.t_rep = nn.Linear(1, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)

        self.input_size = sparse_x_len * emb_dim
        self.num_experts = num_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.tasks = tasks
        self.emb_dim = emb_dim

        self.tower_input_size = self.experts_out
        self.towers_hidden = towers_hidden

        self.softmax = nn.Softmax(dim=1)

        self.c_experts = nn.ModuleList(
            [Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_experts)])
        self.c_w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(self.input_size, num_experts), requires_grad=True) for i in range(self.tasks)])
        self.c_towers = nn.ModuleList([Tower(self.tower_input_size, 2, self.towers_hidden) for i in range(self.tasks)])
        self.c_logit = nn.Linear(2, 1, True)

        self.u_experts = nn.ModuleList(
            [Expert(self.emb_dim, self.emb_dim // 2, self.emb_dim) for i in range(self.num_experts)])
        self.u_w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(self.emb_dim, num_experts), requires_grad=True) for i in range(self.tasks)])
        self.u_towers = nn.ModuleList([Tower(self.emb_dim // 2, 2, self.emb_dim // 4) for i in range(self.tasks)])
        self.t_logit = nn.Linear(2, 1, True)
        self.u_tau = nn.Linear(2, 1, True)

    def self_attn(self, q, k, v):
        Q, K, V = self.Q_w(q), self.K_w(k), self.V_w(v)
        attn_weights = Q.matmul(torch.transpose(K, 1, 2)) / (K.shape[-1] ** 0.5)
        attn_weights = self.softmax(torch.sigmoid(attn_weights))

        outputs = attn_weights.matmul(V)

        return outputs, attn_weights

    def interaction_attn(self, t, x):
        attention = []
        for i in range(self.sparse_x_len):
            temp = self.att_embed_3(torch.relu(
                torch.sigmoid(self.att_embed_1(t)) +
                torch.sigmoid(self.att_embed_2(x[:, i, :]))))
            attention.append(temp)
        attention = torch.squeeze(torch.stack(attention, 1), 2)
        # print('interaction attention', attention)
        attention = torch.softmax(attention, 1)
        # print('mean interaction attention', torch.mean(attention, 0))

        outputs = torch.squeeze(torch.matmul(torch.unsqueeze(attention, 1), x), 1)
        return outputs, attention

    def forward(self, x_input, rebate_amount):
        x_input = x_input.int()
        rebate_amount = rebate_amount.float()
        x_rep = self.emb_resp(x_input)

        # control Multitask
        dims = x_rep.size()  ## batch_size * sparse_x_len * emb_size
        _x_rep = x_rep / torch.linalg.norm(x_rep, dim=1, keepdim=True)
        outputs, attn_weights = self.self_attn(_x_rep, _x_rep, _x_rep)
        _x_rep = torch.reshape(outputs, (dims[0], dims[1] * dims[2]))  ## (batch_size, sparse_x_len * emb_size)

        experts_o = [e(_x_rep) for e in self.c_experts]  ## num_experts , (batch_size, expert_out)
        experts_o_tensor = torch.stack(experts_o)

        gates_o = [self.softmax(_x_rep @ g) for g in self.c_w_gates]  ## gates, (batch_size, num_experts)

        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor for g in
                       gates_o]  ## (num_experts, batch_size, experts_out)
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]  ## (batch_size, experts_out)

        c_outputs = [t(ti) for t, ti in zip(self.c_towers, tower_input)]

        c_logits = []
        c_probs = []
        for output in c_outputs:
            logit = self.c_logit(output)
            c_logits.append(logit)
            c_probs.append(torch.sigmoid(logit))

            # uplift Mutitask
        t_rep = self.t_rep(rebate_amount.reshape(-1, 1))
        _t_rep, _t_attn_weight = self.interaction_attn(t_rep, x_rep)

        u_experts_o = [e(_t_rep) for e in self.u_experts]
        u_experts_o_tensor = torch.stack(u_experts_o)
        u_gates_o = [self.softmax(_t_rep @ g) for g in self.u_w_gates]

        u_tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.emb_dim // 2) * u_experts_o_tensor for g in u_gates_o]
        u_tower_input = [torch.sum(ti, dim=0) for ti in u_tower_input]
        t_outputs = [t(ti) for t, ti in zip(self.u_towers, u_tower_input)]

        u_taus = []
        probs = []
        for output in t_outputs:
            u_taus.append(self.u_tau(output))

        for control_logit, uplift_logit in zip(c_logits, u_taus):
            probs.append(torch.sigmoid(control_logit + uplift_logit))

        return torch.stack(c_logits).squeeze(-1).T, torch.stack(c_probs).squeeze(-1).T, torch.stack(probs).squeeze(
            -1).T, torch.stack(u_taus).squeeze(-1).T

    def calculate_loss(self, is_rev_list, is_pic_list, is_treat,
                       c_logits, c_probs, probs, u_taus):
        # regression rev
        ## rev
        c_logit_fix0 = (c_logits.T)[0].detach()
        uc0 = (c_logits.T)[0]
        ut0 = (c_logit_fix0 + (u_taus.T)[0])

        c_logit_fix1 = (c_logits.T)[1].detach()
        uc1 = (c_logits.T)[1]
        ut1 = (c_logit_fix1 + (u_taus.T)[1])

        rev_true = is_rev_list
        pic_true = is_pic_list
        t_true = is_treat

        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        tt = (1 - t_true) * uc1 + t_true * ut1

        pic_loss1 = criterion((1 - t_true) * uc1 + t_true * ut1, pic_true)
        rev_loss1 = criterion((1 - t_true) * uc0 + t_true * ut0, rev_true)

        if self.enable_drm:
            pic_ite_score = -self.alpha * (probs[1] - c_probs[1])
            rev_ite_score = -self.alpha * (probs[0] - c_probs[1])
            pic_loss_drm = drm(pic_true, 1 - t_true, pic_ite_score)
            rev_loss_drm = drm(rev_true, 1 - t_true, rev_ite_score)
            pic_loss1 = pic_loss1 * self.base_coef + pic_loss_drm * self.drm_coef
            rev_loss1 = rev_loss1 * self.base_coef + rev_loss_drm * self.drm_coef

        loss = pic_loss1 * 0.6 + rev_loss1 * 0.4

        return loss