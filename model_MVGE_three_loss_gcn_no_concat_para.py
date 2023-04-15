import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from sklearn.metrics import *

'''还原图与节点的真实分布+还原agg_feature的属性'''
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, num_layers):
        super(GCN, self).__init__()
        n_h = out_ft
        self.layers = []
        self.num_layers = num_layers
        self.layers.append(GCNConv(in_ft, n_h).cuda())
        for __ in range(num_layers - 1):
            self.layers.append(GCNConv(n_h, n_h).cuda())

    def forward(self, feat, edge_index):
        h_1 = self.layers[0](feat, edge_index)
        for idx in range(self.num_layers - 1):
            h_1 = self.layers[idx + 1](h_1, edge_index)
        return h_1
    
    
class LinearEncoder(nn.Module):
    def __init__(self, in_channels_self,in_channels_agg, out_channels):
        super(LinearEncoder, self).__init__()
        self.linear_in_self = nn.Linear(in_channels_self, out_channels*2)
        self.linear_out_self = nn.Linear(in_channels_self+out_channels*2,out_channels)

        self.gcn1 = GCNConv(in_channels_agg,out_channels*2)
        self.gcn2 = GCNConv(out_channels*2,out_channels)
        self.gcn3 = GCNConv(out_channels,out_channels)
        
        self.linear_out = nn.Linear(in_channels_agg+out_channels*3,out_channels)

    def forward(self,x_self,x_neighbor,edge_index):
        #nn.ReLU()

        l1 = F.relu(self.linear_in_self(x_self))
        l1 = torch.cat((x_self,l1),1)
        l1 = self.linear_out_self(l1)
        

            
        g1 = self.gcn1(x_neighbor,edge_index)
        g2 = self.gcn2(g1,edge_index)
#         g3 = self.gcn3(g2,edge_index)
        
        #concat拼接
        x2 = torch.cat((x_neighbor,g1,g2),1)
        x2 = self.linear_out(x2)
        return l1,x2
    

    
EPS = 1e-15
MAX_LOGSTD = 10    

class GAE(torch.nn.Module):

    def __init__(self, encoder, decoder=None):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder


    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)


    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)


    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):


        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss


    def test(self, z, pos_edge_index, neg_edge_index):

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)
        
        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)



'''与原始VGAE相比，encoder多返回了一个z_neighbor，kl_loss多了一个单节点还原的loss'''
class VGAE(GAE):

    def __init__(self, encoder, decoder=None):
        super(VGAE, self).__init__(encoder, decoder)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu


    def encode(self, *args, **kwargs):
        """"""
        self.__mu__1, self.__logstd__1,z_neighbor = self.encoder(*args, **kwargs)
        self.__logstd__1 = self.__logstd__1.clamp(max=MAX_LOGSTD)
        z_self = self.reparametrize(self.__mu__1, self.__logstd__1)
        
        return z_self,z_neighbor
    
    
    def kl_loss(self, mu=None, logstd=None):

        mu_1 = self.__mu__1
        logstd_1 = self.__logstd__1
        
        loss1 = -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd_1 - mu_1**2 - logstd_1.exp()**2, dim=1))

        return loss1



class MVGE1(GAE):
    def __init__(self, in_channels_self,in_channels_agg,out_channels):
        super(MVGE1, self).__init__(encoder=LinearEncoder(in_channels_self,
                                                            in_channels_agg,
                                                          out_channels),
                                       decoder=InnerProductDecoder())
        
        self.decoder_self = nn.Linear(out_channels,in_channels_self)
        self.decoder_neighbor = nn.Linear(out_channels,in_channels_agg)

    def forward(self, x_self, x_neighbor):
        z_self = self.encode(x_self, x_neighbor,pos_edge_index)
        adj_pred = self.decoder.forward_all(z_self)
        return adj_pred
    
    def embed(self,x_self,x_neighbor,edge_index):
        z1,z2=self.encode(x_self,x_neighbor,edge_index)
        z = torch.cat((z1,z2),1)
        return z,z1,z2
    
    def loss(self, x_self, x_neighbor, pos_edge_index, all_edge_index,a,b):
        z1,z2 = self.encode(x_self,x_neighbor,pos_edge_index)
        z_self = torch.cat((z1,z2),1)

        pos_loss = -torch.log(
            self.decoder(z_self, pos_edge_index, sigmoid=True) + 1e-15).mean()

        # Do not include self-loops in negative samples
        all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
        all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp)

        neg_edge_index = negative_sampling(all_edge_index_tmp, z_self.size(0), pos_edge_index.size(1))
        
        neg_loss = -torch.log(1 - self.decoder(z_self, neg_edge_index, sigmoid=True) + 1e-15).mean()
        
        z_self = self.decoder_self(z1)
        loss1 = F.kl_div(F.log_softmax(z_self,dim=1),x_self,reduction='mean')
        
        z_neighbor = self.decoder_neighbor(z2)
        loss2 = F.kl_div(F.log_softmax(z_neighbor,dim=1),x_neighbor,reduction='mean')
        
        
        # return loss1 + loss2 + (neg_loss+pos_loss)*0.01
        return b*(a*loss1 + (1-a)*loss2) + (1-b)*(neg_loss+pos_loss)*0.01
    


    def single_test(self, x_self,x_neighbor, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
        with torch.no_grad():
            z,z1,z2 = self.embed(x_self,x_neighbor,train_pos_edge_index)

        roc_auc_score, average_precision_score = self.test(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score

    
    def trans_single_node(self,z):
        return torch.mean(torch.stack([z[0:z.shape[0]-2:3],z[1:z.shape[0]-1:3],z[2:z.shape[0]:3]]),0)
    
class MVGE2(GAE):
    def __init__(self, in_channels_self,in_channels_agg,out_channels):
        super(MVGE2, self).__init__(encoder=LinearEncoder(in_channels_self,
                                                            in_channels_agg,
                                                          out_channels),
                                       decoder=InnerProductDecoder())
        
        self.decoder_self = nn.Linear(out_channels,in_channels_self)
        self.decoder_neighbor = nn.Linear(out_channels,in_channels_agg)

    def forward(self, x_self, x_neighbor):
        z_self = self.encode(x_self, x_neighbor,pos_edge_index)
        adj_pred = self.decoder.forward_all(z_self)
        return adj_pred
    
    def embed(self,x_self,x_neighbor,edge_index):
        z1,z2=self.encode(x_self,x_neighbor,edge_index)
        z = torch.cat((z1,z2),1)
        return z,z1,z2
    
    def loss(self, x_self, x_neighbor, pos_edge_index, all_edge_index,a,b):
        z1,z2 = self.encode(x_self,x_neighbor,pos_edge_index)
        z_self = torch.cat((z1,z2),1)

        pos_loss = -torch.log(
            self.decoder(z_self, pos_edge_index, sigmoid=True) + 1e-15).mean()

        # Do not include self-loops in negative samples
        all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
        all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp)

        neg_edge_index = negative_sampling(all_edge_index_tmp, z_self.size(0), pos_edge_index.size(1))
        
        neg_loss = -torch.log(1 - self.decoder(z_self, neg_edge_index, sigmoid=True) + 1e-15).mean()
        
        z_self = self.decoder_self(z1)
        loss1 = F.kl_div(F.log_softmax(z_self,dim=1),x_self,reduction='mean')
        
        z_neighbor = self.decoder_neighbor(z2)
        loss2 = F.kl_div(F.log_softmax(z_neighbor,dim=1),x_neighbor,reduction='mean')
        
        
        return loss1 + loss2 + (neg_loss+pos_loss)*0.01
    


    def single_test(self, x_self,x_neighbor, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
        with torch.no_grad():
            z,z1,z2 = self.embed(x_self,x_neighbor,train_pos_edge_index)

        roc_auc_score, average_precision_score = self.test(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score

    
    def trans_single_node(self,z):
        return torch.mean(torch.stack([z[0:z.shape[0]-2:3],z[1:z.shape[0]-1:3],z[2:z.shape[0]:3]]),0)