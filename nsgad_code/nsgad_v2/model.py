import scipy
import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as sm

def seq_view(seq,title,metric):
    feat_view(seq[:50,-1:,:].squeeze().detach().cpu(),metric,title)
def feat_view(input, metric,title):
    data = sm.pairwise_distances(input, metric=metric)  # 'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'

    cmap = sns.light_palette("red", as_cmap=True)
    sns.heatmap(data, annot=False, cmap=cmap)
    plt.xlabel(title)
    plt.ylabel("Y-axis")
    plt.savefig(f"%s.png"%(title))
    plt.show()
    # plt.close()

class PolyConvFrame(nn.Module):
    def __init__(self,
                conv_fn,
                 n_feats,
                 h_feats,
                depth: int =3,
                alpha: float = 1.0,
                fixed: float = False):
        super().__init__()
        self.depth = depth
        self.basealpha = alpha
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(float(min(1 / alpha, 1))),
                         requires_grad=not fixed) for i in range(depth + 1)
        ])

        self.adj = None
        self.conv_fn = conv_fn
        self.lin1=nn.Linear(n_feats,h_feats)
        self.conbin = nn.Linear(h_feats * len(self.alphas), h_feats)
    def forward(self, feat, adj):
        #应该是一个拉普拉斯矩阵

        n_node = feat.shape[0]
        adj_I = torch.tile(torch.eye(5).reshape(1, 5, 5), (n_node, 1, 1)).to(device=feat.device)
        adj=adj-adj_I

        rowsum = torch.sum(adj, dim=1)
        D_invsqrt = torch.pow(rowsum, -0.5)
        D_invsqrt[torch.isinf(D_invsqrt)] = 0.
        D_invsqrt_matrix = torch.stack([torch.diag(row) for row in D_invsqrt])
        D_invsqrt_split = torch.stack([row.view(-1, 1) for row in D_invsqrt])


        # self.adj =D_invsqrt_matrix@adj@D_invsqrt_matrix
        self.adj =adj_I-D_invsqrt_matrix @ adj @ D_invsqrt_matrix
        # self.adj = (adj @ (adj_I * D_invsqrt_split)) * D_invsqrt_split + adj_I
        feat=self.lin1(feat)
        alphas = [self.basealpha * torch.tanh(_) for _ in self.alphas]#将一个正数映射到（0，1）
        xs = [self.conv_fn(0, [feat], self.adj, alphas)]
        for L in range(1, self.depth + 1):
            tx = self.conv_fn(L, xs, self.adj, alphas)
            xs.append(tx)
        # xs = [x.unsqueeze(1) for x in xs]
        # x = torch.cat(xs, dim=1)
        x=self.conbin(torch.cat(xs, dim=2))

        return x

def JacobiConv(L, xs, adj, alphas, a=2.0, b=-0.25, l=-1.0, r=1.0):
    # 'a': 2.0,
    # 'alpha': 0.5,
    # 'b': -0.25,
    '''
    Jacobi Bases. Please refer to our paper for the form of the bases.
    '''
    if L == 0: return xs[0]
    if L == 1:
        coef1 = (a - b) / 2 - (a + b + 2) / 2 * (l + r) / (r - l)
        coef1 *= alphas[0]
        coef2 = (a + b + 2) / (r - l)
        coef2 *= alphas[0]
        return coef1 * xs[-1] + coef2 * (adj @ xs[-1])
    coef_l = 2 * L * (L + a + b) * (2 * L - 2 + a + b)
    coef_lm1_1 = (2 * L + a + b - 1) * (2 * L + a + b) * (2 * L + a + b - 2)
    coef_lm1_2 = (2 * L + a + b - 1) * (a**2 - b**2)
    coef_lm2 = 2 * (L - 1 + a) * (L - 1 + b) * (2 * L + a + b)
    tmp1 = alphas[L - 1] * (coef_lm1_1 / coef_l)
    tmp2 = alphas[L - 1] * (coef_lm1_2 / coef_l)
    tmp3 = alphas[L - 1] * alphas[L - 2] * (coef_lm2 / coef_l)
    tmp1_2 = tmp1 * (2 / (r - l))
    tmp2_2 = tmp1 * ((r + l) / (r - l)) + tmp2
    # print("系数：",tmp1_2,tmp2_2,tmp3)
    nx = tmp1_2 * (adj @ xs[-1]) - tmp2_2 * xs[-1]
    nx -= tmp3 * xs[-2]
    return nx

class My_BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes,  d=1, batch=False):
        super(My_BWGNN, self).__init__()
        # self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = []
        print(self.thetas)
        for i in range(len(self.thetas)):
            if not batch:
                self.conv.append(PolyConv(h_feats, h_feats, self.thetas[i], lin=False))#可替换的滤波器
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.d = d

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)
    def forward(self, feat,adj):

        # hs=[]
        # embs=[]
        h = self.linear(feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([feat.shape[0],feat.shape[1],0]).to(feat.device)
        size=feat.shape[1]
        adj_I = torch.tile(torch.eye(size).reshape(1, size, size), (feat.shape[0], 1, 1)).to(device=feat.device)
        adj=adj-adj_I

        rowsum = torch.sum(adj, dim=1)
        D_invsqrt = torch.pow(rowsum, -0.5)
        D_invsqrt[torch.isinf(D_invsqrt)] = 0.
        D_invsqrt_matrix = torch.stack([torch.diag(row) for row in D_invsqrt])
        # D_invsqrt_split = torch.stack([row.view(-1, 1) for row in D_invsqrt])
        Lap=adj_I-D_invsqrt_matrix@adj@D_invsqrt_matrix
        for conv in self.conv:
            # 改变 GCN 的聚合方式，设计自己的带通滤波器
            h0 = conv(Lap, h)
            h_final = torch.cat((h_final, h0), dim=2)
            # print(h_final.shape)
        h = self.linear3(h_final)
        h_embedding = self.act(h)
        # h_embedding = h
        # h = self.linear4(h_embedding)
        # hs.append(h)
        # embs.append(h_embedding)

        # h=torch.stack(tuple(hs),dim=0)
        # h_embedding=torch.stack(tuple(embs),dim=0)
        return h_embedding

#
def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d + 1):
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) / (scipy.special.beta(i + 1, d + 1 - i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d + 1):
            inv_coeff.append(float(coeff[d - i]))
        thetas.append(inv_coeff)
    # return [[2,-1]]
    return thetas
class PolyConv(nn.Module):
    def __init__(self,
             in_feats,
             out_feats,
             theta,
             activation=F.leaky_relu,
             lin=False,
             bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin

        # self.reset_parameters()
        # self.linear2 = nn.Linear(out_feats, out_feats, bias)

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)
    def forward(self,L,feat):
        def unlap(L,feat):
            # rowsum = torch.sum(A, dim=1)
            # D_invsqrt=torch.pow(rowsum,-0.5)
            # D_invsqrt[torch.isinf(D_invsqrt)]=0.
            # # D_invsqrt_matrix = torch.stack([torch.diag(row) for row in D_invsqrt])
            # D_invsqrt_split=torch.stack([row.view(-1,1) for row in D_invsqrt])
            return L@feat#I-L
        h = self._theta[0] * feat

        for k in range(self._k):
            feat = L@feat
            h += self._theta[k] * feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)

class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq,1).values

class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values

class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0,2,1)
        sim = torch.matmul(seq,query)
        sim = F.softmax(sim,dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq,sim)
        out = torch.sum(out,1)
        return out

class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.s_a = nn.Sequential(
            # nn.Linear(n_h,  int(n_h/2)),
            # # nn.ReLU(),
            # # nn.Dropout(p=0.5),
            # nn.Linear(int(n_h/2), 1)
            # nn.PReLU(),
            # nn.ReLU(),
            nn.Linear(n_h, int(n_h/2)),
            # nn.Dropout(p=0.7),
            nn.ReLU(),
            nn.Linear(int(n_h/2),int(n_h/4)),
            nn.ReLU(),
            nn.Linear(int(n_h / 4), 1)
            # nn.Sigmoid()
        )

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Sequential):
            for m in self.s_a.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight.data)
                    nn.init.constant_(m.bias, 0.0)
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl,sub_flag):
        scs = []  #
        # positive
        if sub_flag == 1:
            temp1 = h_pl - c
            # temp1 = torch.cat((h_pl,c),dim=1)
            scs.append(self.s_a(temp1))  # 减法聚合之后在经过一次线性层

            # negative
            c_mi = c
            for _ in range(self.negsamp_round):
                c_mi = torch.cat((c_mi[-1:, :], c_mi[:-1, :]), 0)  # 负采样过程
                temp2 = h_pl - c_mi
                # temp2 = torch.cat((h_pl, c_mi), dim=1)
                scs.append(self.s_a(temp2))  # 负样本的减法聚合
            # h_trans=c-h_pl
        elif sub_flag == 0:
            scs.append(self.f_k(h_pl, c))

            # negative
            c_mi = c
            for _ in range(self.negsamp_round):
                c_mi = torch.cat((c_mi[-1:, :], c_mi[:-1, :]), 0)  # 负采样过程
                scs.append(self.f_k(h_pl, c_mi))  # h_pl 指的是未经过聚合的信息，由原节点1433特征经过普通的线性变化得到64
            # h_trans=c-h_pl

        logits = torch.cat(tuple(scs))


        # logits = torch.cat(tuple(scs))

        return logits

class Model_gcn(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout,conv_fn):
        super(Model_gcn, self).__init__()
        self.read_mode = readout
        self.gcn = GCN(n_in, n_h, activation)
        # self.bwg = My_BWGNN(n_in, n_h, num_classes=2, d=2, batch=False)
        # self.specg = PolyConvFrame(conv_fn=conv_fn, n_feats=n_in, h_feats=n_h, depth=5, alpha=0.5, fixed=False)
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc = Discriminator(n_h, negsamp_round)

    def forward(self, seq1, adj, spec_flag,sub_flag,sparse=False):
        if spec_flag=='gcn':
            h_1 = self.gcn(seq1, adj, sparse)
        if spec_flag=='bwg':
            h_1 = self.bwg(seq1,adj)
        if self.read_mode != 'weighted_sum':
            c = self.read(h_1[:,: -1,:])
            h_mv = h_1[:,-1,:]
        else:
            h_mv = h_1[:, -1, :]
            c = self.read(h_1[:,: -1,:], h_1[:,-2: -1, :])

        ret = self.disc(c, h_mv,sub_flag)

        return ret
class Model_bwg(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout,conv_fn):
        super(Model_bwg, self).__init__()
        self.read_mode = readout
        # self.gcn = GCN(n_in, n_h, activation)
        self.bwg = My_BWGNN(n_in, n_h, num_classes=2, d=2, batch=False)
        # self.specg = PolyConvFrame(conv_fn=conv_fn, n_feats=n_in, h_feats=n_h, depth=5, alpha=0.5, fixed=False)
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc = Discriminator(n_h, negsamp_round)

    def forward(self, seq1, adj, spec_flag,sub_flag,sparse=False):
        if spec_flag=='gcn':
            h_1 = self.gcn(seq1, adj, sparse)
        if spec_flag=='bwg':
            h_1 = self.bwg(seq1,adj)
        if self.read_mode != 'weighted_sum':
            c = self.read(h_1[:,: -1,:])
            h_mv = h_1[:,-1,:]
        else:
            h_mv = h_1[:, -1, :]
            c = self.read(h_1[:,: -1,:], h_1[:,-2: -1, :])
        # feat_view(seq1.detach().cpu()[:, -1, :][:100], 'euclidean', 'embedings_Before_Train')
        # feat_view(seq1.detach().cpu()[:, -1, :][:100], 'euclidean', 'embedings_Before_Train')
        # feat_view(h_mv.detach().cpu()[:100], 'euclidean', 'embedings_After_Train')
        # feat_view(c.detach().cpu()[:100], 'euclidean', 'Neibor_embedings_After_Train')
        # feat_view((h_mv - c).detach().cpu()[:100], 'euclidean', 'positive_Sub_embedings_After_Train')
        c_mi = c
        c_neg = torch.cat((c_mi[-1:, :], c_mi[:-1, :]), 0)
        # feat_view((h_mv - c_neg).detach().cpu()[:100], 'euclidean', 'Negative_Sub_embedings_After_Train')
        ret = self.disc(c, h_mv,sub_flag)

        return ret