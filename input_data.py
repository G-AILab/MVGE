import networkx as nx
import pandas as pd
import numpy as np
import igraph as ig
from tqdm import tqdm, tqdm_pandas, tqdm_notebook, tqdm_gui
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score
import scipy.sparse as sp
import random
import multiprocessing as mp
from sklearn.metrics import *

import torch
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
import collections


def NX_to_IG(G, directed=False):
    return ig.Graph(len(G),
                    list(zip(*list(zip(*nx.to_edgelist(G)))[:2])),
                    directed=directed)


def read_data(name,multi_label=False):
    try:
        G = nx.read_adjlist("./input/{}/{}_adjlist.txt".format(name, name),
                            delimiter=' ',
                            nodetype=int,
                            create_using=nx.DiGraph())
        G.add_edges_from([i[::-1] for i in list(G.edges())])  # 显式的加入双向边
        G_label = pd.read_pickle("./input/{}/{}_label.pickle".format(name, name))
        G_attr = pd.read_pickle("./input/{}/{}_attr.pickle".format(name, name))
        G_label = G_label['label'].values

        
    except FileNotFoundError:
        G = nx.Graph()
        data_adj = np.loadtxt('./input/{}/{}_A.txt'.format(name,name), delimiter=',').astype(int)-1
        data_tuple = list(map(tuple, data_adj))
        G.add_edges_from(data_tuple)

        labels_all = np.loadtxt('./input/{}/{}_node_labels.txt'.format(name,name), delimiter=',').astype(int)
        G_label = pd.DataFrame()
        G_label['nodes'] = list(range(len(labels_all)))
        G_label['label'] = labels_all
        G_label['label'] = G_label['label'].map(lambda x: [x])

        attr = np.loadtxt('./input/{}/{}_node_attributes.txt'.format(name,name), delimiter=',')
        G_attr = pd.DataFrame()
        G_attr['nodes'] = list(range(len(labels_all)))
        G_attr['fea_0'] = attr
        
    if multi_label==True:
        G_label = np.concatenate(G_label).reshape(len(G_label), -1)

    iG = NX_to_IG(G, False)
#     for i in tqdm(range(iG.vcount())):
#         G.add_edge(i, i)
 

    print("{} Have {} Nodes, {} Edges, {} Attribute, {} Classes".format(
        name, iG.vcount(), iG.ecount(), G_attr.shape[1] - 1,len(np.unique(G_label))))
    return iG, G, G_label, G_attr

'''-----------------------------------------start ： 计算pair之间的最短路径长度-----------------------------------'''
def precompute_dist_data(edge_index, num_nodes, approximate=0):
    '''
    Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
    :return:
    '''
    graph = nx.Graph()
    edge_list = edge_index.transpose(1,0).tolist()
    graph.add_edges_from(edge_list)

    n = num_nodes
    dists_array = np.zeros((n, n))
    # dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=approximate if approximate>0 else None)
    # dists_dict = {c[0]: c[1] for c in dists_dict}
    dists_dict = all_pairs_shortest_path_length_parallel(graph,cutoff=approximate if approximate>0 else None)
    for i, node_i in enumerate(graph.nodes()):
        shortest_dist = dists_dict[node_i]
        for j, node_j in enumerate(graph.nodes()):
            dist = shortest_dist.get(node_j, -1)
            if dist!=-1:
                # dists_array[i, j] = 1 / (dist + 1)
#                 dists_array[node_i, node_j] = 1 / (dist + 1)
                dists_array[node_i, node_j] = dist
    dists_array[dists_array>=4] = 4
    return dists_array

def all_pairs_shortest_path_length_parallel(graph,cutoff=None,num_workers=4):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes)<50:
        num_workers = int(num_workers/4)
    elif len(nodes)<400:
        num_workers = int(num_workers/2)

    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range,
            args=(graph, nodes[int(len(nodes)/num_workers*i):int(len(nodes)/num_workers*(i+1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict

def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict

def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

'''返回函数的上三角矩阵（单向边和自环都去除，为-1，其他label为0，1，2，3'''
def get_dists(mask_G,nclass):
    
    path_length = dict(nx.all_pairs_shortest_path_length(mask_G, cutoff=nclass-1))
    distance = - np.ones((len(mask_G), len(mask_G))).astype(int)
    
    for u, p in path_length.items():
        for v, d in p.items():
            distance[u][v] = d
            
    distance[distance==-1] = distance.max() + 1
    distance = np.triu(distance) 
    return torch.LongTensor(distance) - 1

def sample(labels,k):
    # then sample k other nodes to make sure class balance
    node_pairs = []
    for i in range(0, labels.max()+1):
        tmp = np.array(np.where(labels==i)).transpose()
        indices = np.random.choice(np.arange(len(tmp)),k, replace=False)
        node_pairs.append(tmp[indices])
        
    node_pairs = np.array(node_pairs).reshape(-1, 2).transpose()
    return node_pairs[0], node_pairs[1]

'''
group 1,2 into the same category, 3, 4, 5 separately
designed for 2-layer GCN
'''
# def _get_label(self):
#     path_length = dict(nx.all_pairs_shortest_path_length(self.graph, cutoff=self.nclass))
#     distance = - np.ones((len(self.graph), len(self.graph))).astype(int)
#     for u, p in path_length.items():
#         for v, d in p.items():
#             distance[u][v] = d
#             distance[distance==-1] = distance.max() + 1

#     # group 1, 2 in to one category
#     distance = np.triu(distance)
#     distance[distance==1] = 2
#     self.distance = distance - 1
#     return torch.LongTensor(distance) - 2

'''------------------------------------end ： 计算pair之间的最短路径长度-------------------------------'''


def process_data(G,G_attr,link_pred=False,multi_agg=True):
    if G_attr.shape[1] >= 2000:
        nodes_matrix = G_attr.drop('nodes', axis=1).values
        M = sp.csr_matrix(nodes_matrix)
        U, S, V = sp.linalg.svds(M.asfptype(), 128)
        W = U * S**0.5
        attribute_0 = W / np.linalg.norm(W, axis=1, keepdims=True)
    else:
        W = G_attr.drop('nodes', axis=1).fillna(0).values
        attribute_0 = W / np.linalg.norm(W, axis=1, keepdims=True)
        attribute_0 = np.nan_to_num(attribute_0)
    
    
    x = torch.tensor(np.array(attribute_0,dtype = np.float32))
    edges = pd.DataFrame(G.edges(), columns = ['u','v'])
    edge_index = torch.tensor(edges.values.T)
    data = Data(x=x,edge_index=edge_index)
    
    data = train_test_split_edges(data, 0.05, 0.1)
    data.edge_index = edge_index
    
    print('-----------start aggregate neighbor-----------')
    if link_pred==True:
        edge_index = data.train_pos_edge_index.numpy().T
        mask_G = nx.Graph()
        mask_G.add_nodes_from(G.nodes())
        mask_G.add_edges_from(edge_index)
        G = mask_G
        
    if multi_agg==True:
        data.x_neighbor,data.x = get_multi_agg(attribute_0,G)
    else:
        data.x_neighbor = get_agg(attribute_0,G)
        
    print('-----------process data completed-----------')
    return data

def process_data_GCN(G,G_attr,link_pred=False,multi_agg=True):
    if G_attr.shape[1] >= 2000:
        nodes_matrix = G_attr.drop('nodes', axis=1).values
        M = sp.csr_matrix(nodes_matrix)
        U, S, V = sp.linalg.svds(M.asfptype(), 128)
        W = U * S**0.5
        attribute_0 = W / np.linalg.norm(W, axis=1, keepdims=True)
    else:
        W = G_attr.drop('nodes', axis=1).fillna(0).values
        attribute_0 = W / np.linalg.norm(W, axis=1, keepdims=True)
        attribute_0 = np.nan_to_num(attribute_0)
    
    
    x = torch.tensor(np.array(attribute_0,dtype = np.float32))
    edges = pd.DataFrame(G.edges(), columns = ['u','v'])
    edge_index = torch.tensor(edges.values.T)
    data = Data(x=x,edge_index=edge_index)
    
    data = train_test_split_edges(data, 0.05, 0.1)
    data.edge_index = edge_index
    
    print('-----------start aggregate neighbor-----------')
    if link_pred==True:
        edge_index = data.train_pos_edge_index.numpy().T
        mask_G = nx.Graph()
        mask_G.add_nodes_from(G.nodes())
        mask_G.add_edges_from(edge_index)
        G = mask_G
        

    data.x_neighbor = get_gcn_agg(data.x,edge_index)
        
    print('-----------process data completed-----------')
    return data

def get_multi_agg(attribute_0,mask_G):
    
    num_nodes = attribute_0.shape[0]
    rd = RandomWalker(mask_G)
    feature = []
    k = 0
    i=0
    for u in tqdm_notebook(range(num_nodes)):
        feature3 = np.mean(get_transform(rd._walk,u,attribute_0,3),axis=0)
        feature5 = np.mean(get_transform(rd._walk,u,attribute_0,5),axis=0)
        feature10 = np.mean(get_transform(rd._walk,u,attribute_0,10),axis=0)
        feature.append([u,np.hstack((feature3,feature5,feature10)),attribute_0[u]])
        
    data = pd.DataFrame(feature,columns=['u','e1','self'])
    attr_neighbor = np.concatenate(data['e1'].values).reshape(data.shape[0],-1)
    x_neighbor = torch.from_numpy(attr_neighbor).float()

    attr_self = np.concatenate(data['self'].values).reshape(data.shape[0],-1)
    x_self = torch.from_numpy(attr_self).float()
    
    return x_neighbor,x_self

# def get_multi_agg(attribute_0,mask_G):
    
#     num_nodes = attribute_0.shape[0]
#     rd = RandomWalker(mask_G)
#     feature = []
#     k = 0
#     i=0
#     for u in tqdm_notebook(range(num_nodes)):
#         col = []
#         for t in [3,5,10]:
#             d = get_transform(rd._walk,u,attribute_0,t)
#             feature.append([u,np.mean(d,axis=0),attribute_0[u]])
#         i+=1


#     data = pd.DataFrame(feature,columns=['u','e1','self'])
#     attr_neighbor = np.concatenate(data['e1'].values).reshape(data.shape[0],-1)
#     x_neighbor = torch.from_numpy(attr_neighbor).float()

#     attr_self = np.concatenate(data['self'].values).reshape(data.shape[0],-1)
#     x_self = torch.from_numpy(attr_self).float()
    
#     return x_neighbor,x_self

from torch_geometric.nn import GCNConv
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
    
def get_gcn_agg(attribute,edge_index):
    channels = attribute.shape[1]
    conv1 = GCNConv(channels, 2 * channels, cached=True)
    conv2 = GCNConv(2 * channels, channels, cached=True)
    x_neighbor = conv1(attribute,edge_index)
    x_neighbor = conv2(x_neighbor,edge_index)
    return x_neighbor

def get_data(name):
    iG,G,G_label,G_attr = read_data(name)
    if G_attr.shape[1] >= 2000:
        nodes_matrix = G_attr.drop('nodes', axis=1).values
        M = sp.csr_matrix(nodes_matrix)
        U, S, V = sp.linalg.svds(M.asfptype(), 128)
        W = U * S**0.5
        attribute_0 = W / np.linalg.norm(W, axis=1, keepdims=True)
    else:
        W = G_attr.drop('nodes', axis=1).fillna(0).values
        attribute_0 = W / np.linalg.norm(W, axis=1, keepdims=True)
        attribute_0 = np.nan_to_num(attribute_0)
    
    
    attribute = np.array(attribute_0,dtype = np.float32)
    x = torch.tensor(attribute)
    
    edges = pd.DataFrame(G.edges(), columns = ['u','v'])
    edge_index = torch.tensor(edges.values.T)

    data = Data(x=x,edge_index=edge_index)
    
    return G,G_label,attribute_0,data
    
def get_pair_wise_score(node_embedding,name,test_size):
    iG,G,G_label,G_attr = read_data(name)
    mask_link_positive = pd.read_pickle("./input/pairwise_node/{}_mask_link_positive.pickle".format(name))
    mask_link_negtive = pd.read_pickle("./input/pairwise_node/{}_mask_link_negtive.pickle".format(name))

    num = len(G.edges())
    mask_link_positive = mask_link_positive.T.sample(n=num).reset_index(drop=True)
    mask_link_negtive = mask_link_negtive.sample(n=num).reset_index(drop=True)

    from sklearn.model_selection import train_test_split
    mask_link_positive_train, mask_link_positive_test = train_test_split(mask_link_positive,test_size=test_size)
    mask_link_negtive_train, mask_link_negtive_test = train_test_split(mask_link_negtive,test_size=test_size)
    train = pd.concat([mask_link_positive_train,mask_link_negtive_train],axis=0)
    test = pd.concat([mask_link_positive_test,mask_link_negtive_test],axis=0)

    nodes_first = node_embedding.loc[train[0]].reset_index(drop=True)
    nodes_second = node_embedding.loc[train[1]].reset_index(drop=True)
    train_pred = pd.DataFrame.mul(nodes_first,nodes_second)
    train_label = [1 for i in range(train.shape[0]//2)] + [0 for i in range(train.shape[0]//2)]

    nodes_first = node_embedding.loc[test[0]].reset_index(drop=True)
    nodes_second = node_embedding.loc[test[1]].reset_index(drop=True)
    test_pred = pd.DataFrame.mul(nodes_first,nodes_second)
    test_label = [1 for i in range(test.shape[0]//2)] + [0 for i in range(test.shape[0]//2)]

    from sklearn.linear_model import LogisticRegressionCV
    clf = LogisticRegressionCV(cv=5,Cs=10,max_iter=100,n_jobs=20,verbose=1,scoring='roc_auc')

    clf.fit(train_pred,train_label)
    auc, ap = roc_auc_score(test_label,clf.predict_proba(test_pred)[:,1]),average_precision_score(test_label,clf.predict_proba(test_pred)[:,1])
    return auc,ap
    
    
# def get_pair_wise_score(node_embedding,G_label,name,test_size):
#     G = nx.read_adjlist("../input/{}/{}_adjlist.txt".format(name, name),
#                             delimiter=' ',
#                             nodetype=int,
#                             create_using=nx.DiGraph())
#     edges = pd.DataFrame(G.edges(), columns = ['u','v'])
#     pair_label = list(map(lambda x:1 if G_label[x[0]]==G_label[x[1]] else 0,G.edges()))

#     nodes_first = node_embedding.loc[edges['u']].reset_index(drop=True)
#     nodes_second = node_embedding.loc[edges['v']].reset_index(drop=True)
#     now_val = pd.DataFrame.mul(nodes_first,nodes_second)

#     from sklearn.linear_model import LogisticRegressionCV
#     clf = LogisticRegressionCV(cv=5,Cs=10,max_iter=100,n_jobs=20,verbose=1,scoring='roc_auc')
#     train, valid, train_label, valid_label = train_test_split(now_val,pair_label,test_size=test_size, random_state=2020)
#     clf.fit(train,train_label)
# #     print("Train Shape {} Valid Shape {}".format(train.shape,valid.shape))
#     auc, ap = roc_auc_score(valid_label,clf.predict_proba(valid)[:,1]),average_precision_score(valid_label,clf.predict_proba(valid)[:,1])
# #     print("Validation SET ROC-AUC Score {} & Average Precision Score {}".format(auc,ap))
#     return auc,ap
    

def get_cv_score(emb, G, G_label, clf):
    ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    tot = 0
    for i in clf:  # svc_linear,svc_rbf,
        k, k1 = [], []
        print(i)
        for test_size in tqdm_notebook(ratios):
            train, test, train_label, test_label = train_test_split(
                emb,
                G_label,
#                 G_label['label'].map(lambda x: x[0]).values,
                test_size=1 - test_size)
            try:
                print('try:',train.shape)
                scores_clf = cross_validate(i,
                                            train,
                                            train_label,
                                            cv=5,
                                            scoring=['f1_micro', 'f1_macro'],
                                            n_jobs=10,
                                            verbose=0)
            except:
                print('except:',train.shape)
                scores_clf = cross_validate(i,
                                            train,
                                            train_label,
                                            cv=5,
                                            scoring=['f1_micro', 'f1_macro'],
                                            n_jobs=10,
                                            verbose=0)
            micro = "%0.4f±%0.4f" % (scores_clf['test_f1_micro'].mean(),
                                     scores_clf['test_f1_micro'].std() * 2)
            macro = "%0.4f±%0.4f" % (scores_clf['test_f1_macro'].mean(),
                                     scores_clf['test_f1_macro'].std() * 2)
            k.append([micro, macro])
            i.fit(train.astype(np.float32), train_label.astype(np.float32))
            k1.append([
                f1_score(test_label, i.predict(test.astype(np.float64)), average='micro'),
                f1_score(test_label, i.predict(test.astype(np.float64)), average='macro')
            ])

        tr = pd.DataFrame(k).T

        tr.columns = ['ratio {}'.format(i) for i in ratios]
        tr.index = ['train-micro', 'train-macro']

        display(tr)

    return tr

def get_all_cv_score(emb, G, G_label, clf):
    ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    tot = 0
    for i in clf:  # svc_linear,svc_rbf,
        k = []
        k1 = []
        print(i)
        for test_size in tqdm_notebook(ratios):
            train, test, train_label, test_label = train_test_split(
                emb,
                G_label,
                test_size=1 - test_size)
            try:
#                 print('try:',train.shape)
                scores_clf = cross_validate(i,
                                            train,
                                            train_label,
                                            cv=5,
                                            scoring=['f1_micro', 'f1_macro'],
                                            n_jobs=10,
                                            verbose=0)
            except:
#                 print('except:',train.shape)
                scores_clf = cross_validate(i,
                                            train,
                                            train_label,
                                            cv=5,
                                            scoring=['f1_micro', 'f1_macro'],
                                            n_jobs=10,
                                            verbose=0)
            k.append([scores_clf['test_f1_micro'].mean(),
                      scores_clf['test_f1_micro'].std() * 2,
                      scores_clf['test_f1_macro'].mean(),
                      scores_clf['test_f1_macro'].std() * 2])
    return k


class RandomWalker(object):
    def __init__(self,nxG):
        super(RandomWalker, self).__init__()
        self.G = nxG

    def _walk(self, start_node, length_walk):
        # Simulate a random walk starting from start node.
        walk = [start_node]
        while len(walk) < length_walk:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) == 0:
                break
            k = int(np.floor(np.random.rand()*len(cur_nbrs)))
            walk.append(cur_nbrs[k])
        #路径去重且删除起始节点元素
#         walk = list(set(walk))
#         walk.remove(start_node)
        return walk

    def _simulate_walks(self, length_walk, num_walks):
        # Repeatedly simulate random walks from each node.
        walks = []
        nodes = list(self.G.nodes())
        for walk_iter in (range(num_walks)):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._walk(node, length_walk))
        return walks

def transform_neighbor_korder_shuffle(x, k):
    return random.choices(x, k=k)

def transform_degree(x):
    return list(dict(G.degree(x)).values)

def get_transform(f,n,a,k):
    return a[f(n,k)]

def get_agg(attribute_0,mask_G):
    
    num_nodes = attribute_0.shape[0]
        

    rd = RandomWalker(mask_G)
    feature = []
    k = 0
    i=0
    for u in tqdm_notebook(range(num_nodes)):
        col = []
        #3,5,10
        for t in [3]:
            d = get_transform(rd._walk,u,attribute_0,t)
            feature.append([u,np.mean(d,axis=0)])
        i+=1
    data = pd.DataFrame(feature,columns=['u','e1'])
    attr_neighbor = np.concatenate(data['e1'].values).reshape(data.shape[0],-1)
    
#     edges = pd.DataFrame(edge_index, columns=['u', 'v'])
#     to_agg = pd.DataFrame(
#         attribute_0,
#         columns=[
#             "attribute_{}".format(i) for i in range(attribute_0.shape[1])
#         ]).fillna(0)

#     edges = edges.merge(to_agg.add_prefix("v_").reset_index().rename(columns={'index' : 'v'}),how='left',on='v')
#     v_col = [i for i in edges.columns if 'v_' in i]
#     agg_func = ['mean']
#     attr_neighbor = edges[['u'] + v_col].groupby(['u'])[v_col].agg(agg_func)

#     single_nodes = list(set(mask_G.nodes()) - set(attr_neighbor.index))
#     single_nodes.sort()
#     single_attr = to_agg.iloc[single_nodes,:]


#     attr_neighbor = attr_neighbor.reindex(range(num_nodes))
#     attr_neighbor.iloc[single_nodes,:] = single_attr.values

#     x_neighbor = torch.from_numpy(attr_neighbor.values).float()
    
    x_neighbor = torch.from_numpy(attr_neighbor).float()
    return x_neighbor


'''返回函数的上三角矩阵（单向边和自环都去除，为-1，其他label为0，1，2，3'''
def get_dists(mask_G,nclass):
    
    path_length = dict(nx.all_pairs_shortest_path_length(mask_G, cutoff=nclass-1))
    distance = - np.ones((len(mask_G), len(mask_G))).astype(int)
    
    for u, p in path_length.items():
        for v, d in p.items():
            distance[u][v] = d
            
    distance[distance==-1] = distance.max() + 1
    distance = np.triu(distance) 
    return torch.LongTensor(distance) - 1

def sample(labels,k):
    # then sample k other nodes to make sure class balance
    node_pairs = []
    for i in range(0, labels.max()+1):
        tmp = np.array(np.where(labels==i)).transpose()
        indices = np.random.choice(np.arange(len(tmp)),k, replace=False)
        node_pairs.append(tmp[indices])
        
    node_pairs = np.array(node_pairs).reshape(-1, 2).transpose()
    return node_pairs[0], node_pairs[1]

####-------------------生成pair wise节点对----------------###
def Graph_load_batch(name = 'ENZYMES'):
    print('Loading graph dataset: '+str(name))
    
    #单向边
    
    try:
        G = nx.read_adjlist("./input/{}/{}_adjlist.txt".format(name, name),
                            delimiter=' ',
                            nodetype=int,
                            create_using=nx.DiGraph())
        G.add_edges_from([i[::-1] for i in list(G.edges())])
        G_label = pd.read_pickle("./input/{}/{}_label.pickle".format(name, name))
        labels_all = np.array(G_label['label'])
        
    except OSError:
        G = nx.Graph()
        data_adj = np.loadtxt('./input/{}/{}_A.txt'.format(name,name), delimiter=',').astype(int)
        data_tuple = list(map(tuple, data_adj))
        G.add_edges_from(data_tuple)
        
        labels_all = np.loadtxt('./input/{}/{}_node_labels.txt'.format(name,name), delimiter=',').astype(int)

        
    graphs = []
    if name in ['ENZYMES','PROTEINS','PROTEINS_full']:
        data_graph_indicator = np.loadtxt('./input/{}/{}_graph_indicator.txt'.format(name,name), delimiter=',').astype(int)
        graph_num = data_graph_indicator.max()
        node_list = np.arange(data_graph_indicator.shape[0])
        max_nodes = 0
        for i in range(graph_num):
            nodes = node_list[data_graph_indicator==i+1]
            G_sub = G.subgraph(nodes)
            graphs.append(G_sub)
    else:
        graphs.append(G)

    return graphs,labels_all

# np.savetxt('ENZYMES',pair_wise,fmt='%d')
# arr = np.loadtxt('ENZYMES',dtype=np.int)
def load_graphs(name):
    pair_wise = []
    
    graphs,labels_all = Graph_load_batch(name)
    
    for graph in graphs:
        n = graph.number_of_nodes()
        label = np.zeros((n, n),dtype=int)
        for i,u in enumerate(graph.nodes()):
            print(i,u)
            for j,v in enumerate(graph.nodes()):
                #只取了上三角
                if labels_all[u] == labels_all[v] and v>u:
                    label[i,j] = 1
                    pair_wise.append([u,v])
                if label.sum() > n*n/4:
                    print('finish generate pairs')
                    break
            if label.sum() > n*n/4:
                print('finish generate pairs')
                break
            
            
    pos_pair_wise = torch.tensor(np.array(pair_wise).T)
    
    
    return pos_pair_wise

'''--------------------------------------------cluster-Distance-----------------------------------------------------------'''
'''
将图中所有的节点都进行了划分,将每个cluster中度最大的节点作为其中心节点；
在每个中心节点上做BFS广度优先遍历，为每个cluster得到一个点集，该点集位于该聚类的k跳邻域内，点集中节点到该cluster的距离为1/k；
'''
class ClusteringMachine(object):
    def __init__(self, graph,cluster_number=20):
        self.graph = graph
        self.cluster_number = cluster_number

    def decompose(self):
        print("\nRandom graph clustering started.\n")
        self.random_clustering()
        central_nodes = self.get_central_nodes()
        print("central_nodes:",central_nodes)
        self.shortest_path_to_clusters(central_nodes)
        

        self.dis_matrix = torch.FloatTensor(self.dis_matrix)

        # self.transfer_edges_and_nodes()

    def random_clustering(self):
        self.clusters = [cluster for cluster in range(self.cluster_number)]
        #给每个点分配聚类类别
        self.cluster_membership = {node: random.choice(self.clusters) for node in self.graph.nodes()}


    def general_data_partitioning(self):
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        for cluster in self.clusters:
            subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if self.cluster_membership[node] == cluster])
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] +  [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
        print('Number of nodes in clusters:', {x: len(y) for x,y in self.sg_nodes.items()})

    def get_central_nodes(self):
        """
        set the central node as the node with highest degree in the cluster
        """
        self.general_data_partitioning()
        central_nodes = {}
        for cluster in self.clusters:
            counter = {}
            for node, _ in self.sg_edges[cluster]:
                counter[node] = counter.get(node, 0) + 1
            sorted_counter = sorted(counter.items(), key=lambda x:x[1])
            central_nodes[cluster] = sorted_counter[-1][0]
        return central_nodes

    def transform_depth(self, depth):
        return 1 / depth

    def shortest_path_to_clusters(self, central_nodes, transform=True):
        """
        Do BFS on each central node, then we can get a node set for each cluster
        which is within k-hop neighborhood of the cluster.
        """
        # self.distance = {c:{} for c in self.clusters}
        self.dis_matrix = -np.ones((self.graph.number_of_nodes(), self.cluster_number))
        for cluster in self.clusters:
            node_cur = central_nodes[cluster]
            visited = set([node_cur])
            q = collections.deque([(x, 1) for x in self.graph.neighbors(node_cur)])
            while q:
                #中心节点的一阶邻居
                node_cur, depth = q.popleft()
                if node_cur in visited:
                    continue
                visited.add(node_cur)

                if transform:
                    self.dis_matrix[node_cur][cluster] = self.transform_depth(depth)
                else:
                    self.dis_matrix[node_cur][cluster] = depth
                for node_next in self.graph.neighbors(node_cur):
                    #获取高阶邻居
                    q.append((node_next, depth+1))

        if transform:
            self.dis_matrix[self.dis_matrix==-1] = 0
        else:
            self.dis_matrix[self.dis_matrix==-1] = self.dis_matrix.max() + 2
        return self.dis_matrix