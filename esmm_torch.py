import torch
import torch.nn as nn



class CtrNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, ouput_dim, use_bn=True):
        super(CtrNetwork, self).__init__()
        if use_bn:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(in_features=hidden_dim, out_features=ouput_dim)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=ouput_dim)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        p = self.mlp(inputs)
        return self.sigmoid(p)


class CvrNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_bn=True):
        super(CvrNetwork, self).__init__()
        if use_bn:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=output_dim)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=output_dim)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        p = self.mlp(inputs)
        return self.sigmoid(p)


class ESMM(nn.Module):
    def __init__(self, x_dim, sparse_x_len, emb_dim, hidden_dim, output_dim, use_bn):
        super(ESMM, self).__init__()
        self.input_size = sparse_x_len * emb_dim + 1
        self.emb_resp = nn.Embedding(x_dim, emb_dim)

        self.ctr_network = CtrNetwork(self.input_size, hidden_dim, output_dim, use_bn)
        self.cvr_network = CvrNetwork(self.input_size, hidden_dim, output_dim, use_bn)

    def forward(self, x, r):
        x_input = x.int()
        r = r.float()
        x_rep = self.emb_resp(x_input)
        dims = x_rep.size()

        _x_rep = torch.reshape(x_rep, (-1, dims[1] * dims[2]))  ## (batch_size, sparse_x_len * emb_size)
        dense_x = torch.concat([_x_rep, r.reshape(-1, 1)], 1)


        # Predict pCTR
        p_ctr = self.ctr_network(dense_x)

        # Predict pCVR
        p_cvr = self.cvr_network(dense_x)

        # Predict pCTCVR
        p_ctcvr = torch.mul(p_ctr, p_cvr)
        return  torch.stack([p_ctr, p_ctcvr]).squeeze(-1).T

    def cal_loss(self, is_rev_lists, is_pic_lists, is_rev_sample_weights, is_pic_sample_weights, outputs):
        criterion = nn.BCELoss(reduction='mean')
        loss1 = criterion((outputs.T)[0], is_rev_lists)
        loss2 = criterion((outputs.T)[1], is_pic_lists)
        loss = loss1 * 0.4 + loss2 * 0.6
        return loss
