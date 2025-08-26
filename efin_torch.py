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


class EFIN(nn.Module):
    """
    EFIN class -- a explicit feature interaction network with two heads.
    """

    def __init__(self, 
                 x_dim,
                 sparse_x_len,
                 emb_dim,
                 layer_units=None, 
                 layer_activations=None, 
                 batch_norm=True,
                 enable_drm=False,
                 bn_decay=0.9,
                 base_coef=0.1,
                 drm_coef=10):
        super(EFIN, self).__init__()
        self.sparse_x_len = sparse_x_len
        self.enable_drm = enable_drm
        self.base_coef = base_coef
        self.drm_coef = drm_coef
        self.batch_norm = batch_norm
        self.bn_decay = bn_decay 

        self.att_embed_1 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.att_embed_2 = nn.Linear(emb_dim, emb_dim)
        self.att_embed_3 = nn.Linear(emb_dim, 1, bias=False)

         # Process layer units and activations
        if isinstance(layer_units, str):
            self.layer_units = [int(x) for x in layer_units.split(',')]
        elif isinstance(layer_units, list):
            self.layer_units = layer_units
        else:
            self.layer_units = [64, 32]  # 默认值
            
        if isinstance(layer_activations, str):
            self.layer_activations = [None if x == 'None' else x for x in layer_activations.split(',')]
        elif isinstance(layer_activations, list):
            self.layer_activations = layer_activations
        else:
            self.layer_activations = ['relu', 'relu']  # 默认值


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
      
        self.emb_dim = emb_dim


        self.softmax = nn.Softmax(dim=1)
        

        self.control_mlp = self._build_mlp_network(
            input_size=self.input_size,
            layer_units=self.layer_units,
            layer_activations=self.layer_activations,
            output_size=2
        )

        self.uplift_mlp = self._build_mlp_network(
            input_size=self.emb_dim,
            layer_units=self.layer_units,
            layer_activations=self.layer_activations,
            output_size=2
        )

        self.c_logit = nn.Linear(2, 1, True)
        self.c_tau = nn.Linear(2, 1, True)
        self.t_logit = nn.Linear(2, 1, True)
        self.u_tau = nn.Linear(2, 1, True)

    def _build_mlp_network(self, input_size, layer_units, layer_activations, output_size=2):
        layers = []
        in_features = input_size

        for units, activation in zip(layer_units, layer_activations):
            layers.append(nn.Linear(in_features, units))
             # 添加批归一化
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(units, momentum=1-self.bn_decay))
            
            # 添加激活函数
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
                
            in_features = units
        
        # 添加输出层
        layers.append(nn.Linear(in_features, output_size))
        return nn.Sequential(*layers)
        

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
    
    def CrontrolNet(self, x):
        """处理控制组（无干预）预测
        Args:
            x_indices: [batch_size, sparse_x_len] 特征索引（整数）
            x_values: [batch_size, sparse_x_len] 特征值
        """
        c_output = self.control_mlp(x)
        c_logit = self.c_logit(c_output)
        c_tau = self.c_tau(c_output)
        c_prob = torch.sigmoid(c_logit)
        return c_logit, c_tau, c_prob


    def UpliftNet(self, x):
        """处理干预效果（返现效应）预测
        Args: 
            x : [batch_size, sparse_x_len] 
            r : [batch_size, 1]
        """ 
        t_output = self.uplift_mlp(x)
        t_logit = self.t_logit(t_output)
        u_tau = self.u_tau(t_output)

        t_prob = torch.sigmoid(t_logit)
        return t_logit, u_tau, t_prob 
    

    def forward(self, x_input, rebate_amount):
        x_input = x_input.int()
        rebate_amount = rebate_amount.float()
        x_rep = self.emb_resp(x_input)

        # control Multitask
        dims = x_rep.size()  ## batch_size * sparse_x_len * emb_size
        _x_rep = x_rep / torch.linalg.norm(x_rep, dim=1, keepdim=True)
        outputs, attn_weights = self.self_attn(_x_rep, _x_rep, _x_rep)
        _x_rep = outputs.view(x_input.size(0), -1)  ## (batch_size, sparse_x_len * emb_size)

        c_logit, c_tau, c_prob = self.CrontrolNet(_x_rep)

        t_rep = self.t_rep(rebate_amount.unsqueeze(1))
        
        _t_rep, _t_attn_weight = self.interaction_attn(t_rep, x_rep)
        t_logit, u_tau, t_prob = self.UpliftNet(_t_rep)

        uplift = torch.sigmoid(u_tau)

        final_output = [c_logit, c_tau, c_prob, t_logit, u_tau, t_prob, uplift]

        return torch.stack(final_output).squeeze(-1).T

    def cal_loss(self, y, output, is_treatment):
        # regression rev
        ## rev
        c_logit, c_tau,  c_prob, t_logit, u_tau, t_prob = output[:, 0], output[:, 1], output[:, 2], output[:, 3], output[:, 4], output[:, 5]
        # regression
        c_logit_fix = c_logit.detach()
        uc = c_logit
        ut = (c_logit_fix + u_tau)

        # response loss
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        loss1 = criterion((1 - is_treatment) * uc + is_treatment * ut, y)
        loss2 = criterion(t_logit, 1 - is_treatment)
        loss = loss1 + loss2

        return loss