import os
import torch.nn as nn
from evaluate import evaluate
from embedder import embedder
from utils.process import GCN, update_S, drop_feature, Linearlayer
import numpy as np
from tqdm import tqdm
import random as random
import torch
from typing import Any, Optional, Tuple
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)




class DMG(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.criteria = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        self.log_sigmoid = nn.LogSigmoid()
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)
    def training(self):
        seed = self.args.seed

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # # ===================================================#

        features = [feature.to(self.args.device) for feature in self.features]
        adj_list = [adj.to(self.args.device) for adj in self.adj_list]
        for i in range(self.args.num_view):
            features[i] = drop_feature(features[i], self.args.feature_drop)

        print("Started training...")


        ae_model = GNNDAE(self.args).to(self.args.device)
        # graph independence regularization network
        mea_func = []
        for i in range(self.args.num_view):
            mea_func.append(Measure_F(self.args.c_dim, self.args.p_dim,
                                  [self.args.phi_hidden_size] * self.args.phi_num_layers,
                                  [self.args.phi_hidden_size] * self.args.phi_num_layers).to(self.args.device))
        # Optimizer
        if self.args.num_view == 2:
            optimizer = torch.optim.Adam([
                {'params': mea_func[0].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
                {'params': mea_func[1].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
                {'params': ae_model.parameters(), 'lr': self.args.lr_min}
            ], lr=self.args.lr_min)
        else:
            optimizer = torch.optim.Adam([
                {'params': mea_func[0].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
                {'params': mea_func[1].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
                {'params': mea_func[2].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
                {'params': ae_model.parameters(), 'lr': self.args.lr_min}
            ], lr=self.args.lr_min)

        # model.train()
        ae_model.train()
        mea_func[0].train()
        mea_func[1].train()
        if self.args.num_view == 3:
            mea_func[2].train()
        best = 1e9

        for itr in tqdm(range(1, self.args.num_iters + 1)):

            # Solve the S subproblem
            U = update_S(ae_model, features, adj_list, self.args.c_dim, self.args.device)

            # Update network for multiple epochs
            for innerepoch in range(self.args.inner_epochs):

                # Backprop to update
                loss, match_err, recons, corr, contrastive, common, private = trainmultiplex(ae_model, mea_func, U, features, adj_list, self.idx_p_list, self.args, optimizer, self.args.device, itr*innerepoch)
            if loss < best:
                best = loss
                cnt_wait = 0
            elif loss > best and itr > 100:
                cnt_wait += 1
            if cnt_wait == self.args.patience:
                print("Early stopped!")
                break

            print('====> Iteration: {} Loss = {:.4f}'.format(
                itr, loss))
        if self.args.use_pretrain:
            ae_model.load_state_dict(
                torch.load('saved_model/best_{}_{}.pkl'.format(self.args.dataset, self.args.custom_key)))
        print("Evaluating...")
        ae_model.eval()
        embedding = []
        hf = update_S(ae_model, features, adj_list, self.args.c_dim, self.args.device)
        _, private = ae_model.embed(features, adj_list)
        private = sum(private) / self.args.num_view
        embedding.append(hf)
        embedding.append(private)
        embeddings = torch.cat(embedding,1)
        macro_f1s, micro_f1s = evaluate(embeddings, self.idx_train, self.idx_val, self.idx_test, self.labels,task=self.args.custom_key,epoch = self.args.test_epo,lr = self.args.test_lr,iterater=self.args.iterater) #,seed=seed
        return macro_f1s, micro_f1s

def compute_corr(x1, x2):
    # Subtract the mean
    x1_mean = torch.mean(x1, 0, True)
    x1 = x1 - x1_mean
    x2_mean = torch.mean(x2, 0, True)
    x2 = x2 - x2_mean

    # Compute the cross correlation
    sigma1 = torch.sqrt(torch.mean(x1.pow(2)))
    sigma2 = torch.sqrt(torch.mean(x2.pow(2)))
    corr = torch.abs(torch.mean(x1*x2))/(sigma1*sigma2)

    return corr

# The loss function for matching and reconstruction
def loss_matching_recons(s, x_hat, x, U_batch, idx_p_list, args, epoch):
    l = torch.nn.MSELoss(reduction='sum')

    # Matching loss
    match_err = l(torch.cat(s, 1), U_batch.repeat(1, args.num_view))/s[0].shape[0]
    recons_err = 0
    # Feature reconstruction loss
    for i in range(args.num_view):
        recons_err += l(x_hat[i], x[i])
    recons_err /= s[0].shape[0]

    # Topology reconstruction loss
    interval = int(args.neighbor_num/args.sample_neighbor)
    neighbor_embedding = []
    for i in range(args.num_view):
        neighbor_embedding_0 = []
        for j in range(0, args.sample_neighbor+1):
            neighbor_embedding_0.append(x[i][idx_p_list[i][(epoch + interval * j) % args.neighbor_num]])
        neighbor_embedding.append(sum(neighbor_embedding_0) / args.sample_neighbor)
    recons_nei = 0
    for i in range(args.num_view):
        recons_nei += l(x_hat[i], neighbor_embedding[i])
    recons_nei /= s[0].shape[0]

    return match_err, recons_err, recons_nei


# The loss function for independence regularization
def loss_independence(phi_c_list, psi_p_list):
    # Correlation
    corr = 0
    for i in range(len(phi_c_list)):
        corr += compute_corr(phi_c_list[i], psi_p_list[i])
    return corr


# cContrastive loss
def loss_contrastive(U, private, adj_list, args):
    i = 0
    loss = 0
    for adj in adj_list:
        adj = adj_list[i]
        out_node = adj.to_sparse()._indices()[1]
        random = np.random.randint(out_node.shape[0], size=int((out_node.shape[0] / args.sample_num)))
        sample_edge = adj.to_sparse()._indices().T[random]
        dis = F.cosine_similarity(U[sample_edge.T[0]],U[sample_edge.T[1]])
        a, maxidx = torch.sort(dis, descending=True)
        idx1 = maxidx[:int(sample_edge.shape[0]*0.2)]
        b, minidx = torch.sort(dis, descending=False)
        idx2 = minidx[:int(sample_edge.shape[0]*0.1)]
        private_sample_0 = private[i][sample_edge[idx1].T[0]]
        private_sample_1 = private[i][sample_edge[idx1].T[1]]
        private_sample_2 = private[i][sample_edge[idx2].T[0]]
        private_sample_3 = private[i][sample_edge[idx2].T[1]]
        i += 1
        loss += semi_loss(private_sample_0, private_sample_1, private_sample_2, private_sample_3, args)
    return loss


def semi_loss(z1, z2, z3, z4, args):
    f = lambda x: torch.exp(x / args.tau)
    positive = f(F.cosine_similarity(z1, z2))
    negative = f(F.cosine_similarity(z3, z4))
    return -torch.log(
        positive.sum()
        / (positive.sum() + negative.sum() ))

def trainmultiplex(model, mea_func, U, features, adj_list,idx_p_list, args,  optimizer, device, epoch):

    model.train()
    mea_func[0].train()
    mea_func[1].train()
    if args.num_view == 3:
        mea_func[2].train()
    common, private, recons = model(features, adj_list)
    match_err, recons_err, recons_nei = loss_matching_recons(common, recons, features, U, idx_p_list, args, epoch)
    # Independence regularizer loss
    phi_c_list = []
    psi_p_list = []
    for i in range(args.num_view):
        phi_c, psi_p = mea_func[i](common[i], private[i])
        phi_c_list.append(phi_c)
        psi_p_list.append(psi_p)
    corr = loss_independence(phi_c_list, psi_p_list)
    loss_con = loss_contrastive(U, private, adj_list, args)
    # Compute the overall loss, note that we use the gradient reversal trick
    # and that's why we have a 'minus' for the last term
    loss = match_err + args.alpha*(recons_err+recons_nei) - args.beta* corr + args.lammbda * loss_con

    # Update all the parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, match_err, recons_err + recons_nei, corr, loss_con, common, private




class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

def grad_reverse(x, coeff):
    return GradientReversalLayer.apply(x, coeff)


class GNNEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pipe = GCN(args.ft_size, args.hid_units, args.activation, args.dropout, args.isBias)
        # map to common
        self.S = nn.Linear(args.hid_units, args.c_dim)
        # map to private
        self.P = nn.Linear(args.hid_units, args.p_dim)

    def forward(self, x, adj):
        tmp = self.pipe(x, adj)
        common = self.S(tmp)
        private = self.P(tmp)
        return common, private

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.linear1 = Linearlayer(args.decolayer,args.c_dim + args.p_dim, args.hid_units, args.ft_size)
        self.linear2 = nn.Linear(args.ft_size, args.ft_size)

    def forward(self, s, p):
        recons = self.linear1(torch.cat((s, p), 1))
        recons = self.linear2(F.relu(recons))
        return recons

class GNNDAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_view = self.args.num_view
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for _ in range(self.args.num_view):
            self.encoder.append(GNNEncoder(args))
            self.decoder.append(Decoder(args))

    def encode(self, x, adj_list):
        common = []
        private = []
        for i in range(self.args.num_view):
            tmp = self.encoder[i](x[i], adj_list[i])
            common.append(tmp[0])
            private.append(tmp[1])

        return common, private

    def decode(self, s, p):
        recons = []
        for i in range(self.num_view):
            tmp = self.decoder[i](s[i], p[i])
            recons.append(tmp)

        return recons

    def forward(self, x, adj):
        common, private = self.encode(x, adj)
        recons = self.decode(common, private)

        return common, private, recons

    def embed(self, x, adj_list):
        common = []
        private = []
        for i in range(self.args.num_view):
            tmp = self.encoder[i](x[i], adj_list[i])
            common.append(tmp[0].detach())
            private.append(tmp[1].detach())
        return common, private

class MLP(nn.Module):
    def __init__(self, input_d, structure, output_d, dropprob=0.0):
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropprob)
        struc = [input_d] + structure + [output_d]

        for i in range(len(struc)-1):
            self.net.append(nn.Linear(struc[i], struc[i+1]))

    def forward(self, x):
        for i in range(len(self.net)-1):
            x = F.relu(self.net[i](x))
            x = self.dropout(x)

        # For the last layer
        y = self.net[-1](x)

        return y

#measurable functions \phi and \psi
class Measure_F(nn.Module):
    def __init__(self, view1_dim, view2_dim, phi_size, psi_size, latent_dim=1):
        super(Measure_F, self).__init__()
        self.phi = MLP(view1_dim, phi_size, latent_dim)
        self.psi = MLP(view2_dim, psi_size, latent_dim)
        # gradient reversal layer
        self.grl1 = GradientReversalLayer()
        self.grl2 = GradientReversalLayer()

    def forward(self, x1, x2):
        y1 = self.phi(grad_reverse(x1,1))
        y2 = self.psi(grad_reverse(x2,1))
        return y1, y2







