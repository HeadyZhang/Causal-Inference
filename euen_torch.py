
import torch 
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np 


class CrossNet(nn.Module):
    """轻量级特征交叉网络（DCN-v2）"""
    def __init__(self, input_dim, num_layers=2):
        super(CrossNet, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_layers) 
        ])

    def forward(self, x):
        x0 = x 
        for layer in self.layers: 
            x = x0 * layer(x) + x 
        return x 

class EUEN(nn.Module):
    def __init__(self, 
                 x_dim, 
                 sparse_x_len,
                 emb_dim, 
                 layer_units=None, 
                 layer_activations=None, 
                 lr = 0.001,
                 logger=None, 
                 bn_decay=0.9, 
                 batch_norm=True,
                 use_cross_x=False):
        super(EUEN, self).__init__()
        self.logger = logger 
        self.x_dim = x_dim

        self.sparse_x_len = sparse_x_len 
        self.lr = lr
        self.batch_norm = batch_norm

        self.bn_decay = bn_decay 

        self.emb_dim = emb_dim

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


        # 特征嵌入
        self.emb = nn.Embedding(x_dim,  self.emb_dim)

        # 特征交叉
        self.cross_dim = sparse_x_len * emb_dim 
        self.cross_net = CrossNet(self.cross_dim, num_layers=2)
        self.cross_net_u = CrossNet(self.cross_dim + 1, num_layers=2)
        self.use_cross_x = use_cross_x


        self.control_mlp = self._build_mlp_network(input_size=self.cross_dim, 
                                                   layer_units=self.layer_units,
                                                   layer_activations=self.layer_activations,
                                                   output_size=2)
        
        self.uplift_mlp = self._build_mlp_network(input_size=self.cross_dim + 1, 
                                                  layer_units=self.layer_units,
                                                  layer_activations=self.layer_activations,
                                                  output_size=2)
        

        self.c_logit = nn.Linear(2, 1, True)
        self.c_tau = nn.Linear(2, 1, True)
        self.t_logit = nn.Linear(2, 1, True)
        self.u_tau = nn.Linear(2, 1, True)


    def _build_mlp_network(self, input_size, layer_units, layer_activations, output_size=1):
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
    
    def forward(self, x, r):
        x = x.int()
        r = r.float()

        emb_x = self.emb(x).view(x.size(0), -1)  ## [batch_size, sparse_x_len * emb_size]

        dense_x = torch.concat([emb_x, r.unsqueeze(1)], 1)

        if self.use_cross_x:
            dense_x = self.cross_net_u(dense_x)

        ## Control网络预测
        c_logit, c_tau, c_prob = self.CrontrolNet(emb_x)

        ## Uplift网络预测
        t_logit, u_tau, t_prob = self.UpliftNet(dense_x)

        u_prob = torch.sigmoid(c_logit + u_tau)

        final_output = [c_logit, c_tau, c_prob, t_logit, u_tau, u_prob]

        return torch.stack(final_output).squeeze(-1).T

    
    def cal_loss(self, y, output, treatment):
        c_logit, c_tau,  c_prob, t_logit, u_tau = output[:, 0], output[:, 1], output[:, 2], output[:, 3], output[:, 4]

        c_logit_fix = c_logit.detach()
        uc = c_logit
        ut = (c_logit_fix + u_tau)
        
        # 计算期望结果
        expected_outcome = (1 - treatment) * uc + treatment * ut 
        
        # 计算二元交叉熵损失
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        loss = criterion(expected_outcome, y)

        return loss 





       











