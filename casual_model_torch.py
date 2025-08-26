import torch


# criterion
def ce(p, y, epsilon=1e-6):
    p = torch.clamp(p.squeeze(), min=epsilon, max=1 - epsilon)
    y = y.squeeze()
    return -torch.log(y * p + (1 - y) * (1 - p))


class UpliftModel(torch.nn.Module):
    def __init__(self, x_dim, sparse_x_len, emb_size, layer_units, layer_activations,
                 use_bn=True, model_type='tlearner', base_units=None, base_activations=None, alpha=None):
        super(UpliftModel, self).__init__()
        self.x_dim = x_dim
        self.sparse_x_len = sparse_x_len
        self.emb_size = emb_size
        self.embedding_layer = torch.nn.Embedding(num_embeddings=self.x_dim, embedding_dim=self.emb_size)
        torch.nn.init.xavier_normal_(self.embedding_layer.weight)
        self.use_bn = use_bn
        self.q0 = torch.nn.ModuleList()
        self.q1 = torch.nn.ModuleList()
        self.model_type = model_type
        self.alpha = alpha
        print(f'use model: {self.model_type}')

        prev_unit = self.sparse_x_len * self.emb_size  # 无基础网路设置维度为embedding后维度
        if self.model_type in ('tarnet', 'cfrnet', 'dragonnet'):
            # 增加公共隐层网络
            self.base_nn = torch.nn.ModuleList()
            assert len(base_units) == len(base_activations) > 0
            for i, (unit, activation) in enumerate(zip(base_units, base_activations)):
                self.base_nn.append(torch.nn.Linear(prev_unit, unit, True))
                if self.use_bn:
                    self.base_nn.append(torch.nn.BatchNorm1d(unit))
                prev_unit = unit        # 记录网络输出维度
        if self.model_type in ('cfrnet'):
            try:
                from geomloss.samples_loss import SamplesLoss
            except Exception:
                print('geomloss not installed')
            self.ipm_reg = SamplesLoss(loss='sinkhorn', p=2)        # Wasserstein距离计算
        if self.model_type in ('dragonnet'):
            assert alpha > 0
            # 增加倾向性得分预测头
            self.g_layer = torch.nn.Linear(prev_unit, 1, True)
            self.g_activation = torch.nn.Sigmoid()
        for i, (unit, activation) in enumerate(zip(layer_units, layer_activations)):
            # 干预组/非干预组双模型
            if i == len(layer_units) - 1:
                # 最后一层, 不进行bn
                self.q0.append(torch.nn.Linear(prev_unit, unit, True))
                self.q1.append(torch.nn.Linear(prev_unit, unit, True))
            else:
                self.q0.append(torch.nn.Linear(prev_unit, unit, True))
                self.q1.append(torch.nn.Linear(prev_unit, unit, True))
                if self.use_bn:
                    self.q0.append(torch.nn.BatchNorm1d(unit))
                    self.q1.append(torch.nn.BatchNorm1d(unit))
            prev_unit = unit
            if activation == 'relu':
                self.q0.append(torch.nn.ReLU())
                self.q1.append(torch.nn.ReLU())
            elif activation == 'sigmoid':
                self.q0.append(torch.nn.Sigmoid())
                self.q1.append(torch.nn.Sigmoid())

    def forward(self, x, t):
        # 执行嵌入操作
        other = {}
        dense_x = self.embedding_layer(x).reshape(-1, x.shape[1] * self.emb_size)
        if self.model_type in ('tarnet', 'cfrnet', 'dragonnet'):
            for m in self.base_nn:
                dense_x = m(dense_x)
            if self.model_type in ('cfrnet'):
                # 计算t=0和t=1数据隐层的分布距离
                dense_x_0 = dense_x[t == 0]
                dense_x_1 = dense_x[t == 1]
                other = self.ipm_reg(dense_x_0, dense_x_1)
        elif self.model_type in ('dragonnet'):
            g = self.g_layer(dense_x)
            g = self.g_activation(g)
            other = g
        else:
            other = torch.zeros_like(t)
        p0 = dense_x
        p1 = dense_x
        for m0, m1 in zip(self.q0, self.q1):
            p0 = m0(p0)
            p1 = m1(p1)
        return p0, p1, other

    def cal_loss(self, p0, p1, y, t, other):
        l0 = ce(p0, y) * (1 - t)
        l1 = ce(p1, y) * t
        loss = (l0 + l1).mean()
        if self.model_type in ('dragonnet'):
            loss += self.alpha * ce(other, t).mean()
        if self.model_type in ('cfrnet'):
            loss += self.alpha * other
        return loss
