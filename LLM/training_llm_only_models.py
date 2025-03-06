#from mestrado_v3.notebooks.llm_functions import *

import pandas as pd
import numpy as np
import matplotlib as plt
import os, errno
import shutil
import networkx as nx
from networkx.algorithms import bipartite
import json
#import yaml
import ast
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors
import pickle
import time
from random import randint

#import tensorflow as tf

import torch
import torch_geometric
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import SAGEConv, GATConv, Linear, to_hetero
from torch_geometric.data import HeteroData

os.environ['TORCH'] = torch.__version__
print(torch.__version__)
print(torch.version.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# keyphrases= ["keyphrase2","keyphrase3","keyphrase23"]
# sample_size = 100
# iterations = range(10)
# # datasets = ["Dmoz_Computers","Dmoz_Health","Dmoz_Science","Dmoz_Sports","Industry_Sector","NSF","classic4","re8","review_polarity","SyskillWebert","webkb_parsed"]#  "CSTR"
# # n_rotulated = [1,5,10,20,30]
# datasets = ["CSTR"]
# n_rotulated = [1,5,10,20]

# output_datasets = "output_datasets_100"

keyphrases= ["keyphrase2","keyphrase3","keyphrase23"]
iterations = range(10)
datasets = ["Dmoz_Sports","Industry_Sector","NSF","classic4","re8","review_polarity","SyskillWebert","webkb_parsed"]#  Dmoz_Sports
# já foi:
#datasets = ["CSTR","Dmoz_Computers","Dmoz_Health","Dmoz_Science",]
n_rotulated = ["LLM_only"]
output_datasets = ["output_datasets_10","output_datasets_50","output_datasets_100"]
output_datasets_dict = {"output_datasets_10":10,"output_datasets_50":50,"output_datasets_100":100}

# try:
#     os.makedirs("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/")
# except OSError as e:
#     if e.errno != errno.EEXIST:
#         raise

# df_list = []

# for i in range(10):
#     aux_list = []

#     aux_list.append(pd.DataFrame(columns = ['1','5','10','20','30'])) # acuracia
#     aux_list.append(pd.DataFrame(columns = ['1','5','10','20','30'])) # precision
#     aux_list.append(pd.DataFrame(columns = ['1','5','10','20','30'])) # recall
#     aux_list.append(pd.DataFrame(columns = ['1','5','10','20','30'])) # f1-score
#     aux_list.append(pd.DataFrame(columns = ['1','5','10','20','30'])) # support
#     aux_list.append(pd.DataFrame(columns = ['1','5','10','20','30'])) # time

#     df_list.append(aux_list)

#     try:
#         os.makedirs("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(i))
#     except OSError as e:
#         if e.errno != errno.EEXIST:
#             raise

#     df_list[i][0].to_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(i)+"/df_acc.pkl") # acuracia
#     df_list[i][1].to_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(i)+"/df_pre.pkl") # precision
#     df_list[i][2].to_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(i)+"/df_rec.pkl") # recall
#     df_list[i][3].to_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(i)+"/df_f1.pkl")  # f1-score
#     df_list[i][4].to_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(i)+"/df_sup.pkl") # support
#     df_list[i][5].to_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(i)+"/df_time.pkl") # time

def set_up_graph(model_name,rotulated,iteration):
    # carregamento de nós do grafo
    with open('/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/processed_datasets/'+dataset_name+'/'+keyphrase+'/graph.pkl','rb') as f:
        aux =  pickle.load(f)
        document_nodes, context_nodes, edges, real_y_values, class_number, classes = aux[0], aux[1], aux[2], aux[3], aux[4], aux[5]

    # carrega mascaras antigas
    # with open('/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/processed_datasets/'+dataset_name+'/fake_Y/fake_Y_'+str(rotulated)+'_'+str(iteration)+'.pkl', 'rb') as f:
    with open('/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/'+output_dataset+'/'+dataset_name+'/new_masks_and_y_'+str(sample_size)+'/mask_rot'+str(rotulated)+'_'+str(iteration)+'.pkl', 'rb') as f:
        aux =  pickle.load(f)
        fake_y = aux[3]

        for i,y in enumerate(fake_y):
            if y == class_number:
                fake_y[i] = randint(0, (class_number-1))


    # Define objeto do grafo
    data = HeteroData()
    # nodes
    data['document'].x = document_nodes
    data['concept'].x = context_nodes
    # edges
    data['document', 'has', 'concept'].edge_index = edges
    # class labels
    data['document'].y = fake_y
    # Setting graph to undirected
    data = T.ToUndirected()(data)
    # Remove duplicate edges
    data = T.RemoveDuplicatedEdges()(data)
    # Ensure date in using gpu
    data = data.to(device)
    # print
    print(data)

    return data, real_y_values, class_number, classes

def train(model,optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['document'].train_mask
    loss = F.cross_entropy(out['document'][mask], data['document'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)

def test(model,real_y_values):
    # pegas as predições
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    pred = out['document'].argmax(dim=1)

    # resgata o y verdadeiro
    real_y = torch.tensor(real_y_values).to(device)

    accs = []
    for mask in [data['document'].train_mask, data['document'].test_mask]:
        accs.append(int((pred[mask] == real_y[mask]).sum()) / int(mask.sum()))
    return accs


def get_y(model,real_y_values):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    pred = out['document'].argmax(dim=1)
    print("PREDICTIONS ->", pred)

    # resgata o y verdadeiro
    real_y = torch.tensor(real_y_values).to(device)

    with open('/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/processed_datasets/'+dataset_name+'/masks/mask_rot'+str(rotulated)+'_'+str(iteration)+'.pkl', 'rb') as f:
        aux = pickle.load(f)
        train_mask, val_mask, test_mask = aux[0],aux[1],aux[2]

    # y_pred, y_true
    # return pred[data['document'].test_mask], real_y
    # Retorna as predições para todos os dados de teste incluindo os rotulados pela LLM
    return pred[test_mask], real_y[test_mask]

# First Network
from torch_geometric.nn import GraphConv

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GraphConv(-1, hidden_channels)
        self.conv2 = GraphConv(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
# Second Network
from torch_geometric.nn import GATv2Conv

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATv2Conv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x
    
def training_iteration(i, rotulated, GCN_or_GAT, data, real_y_values, class_number,classes):
    # Define classes
    class_number = class_number
    target_names = classes

    # for i in range(10):
    print(f"\n===============================================")
    print(f"================ {GCN_or_GAT} - MODEL {i} ================")
    print(f"===============================================\n")

    # with open('/content/drive/MyDrive/Mestrado_Grafos/processed_datasets/'+dataset_name+'/masks/mask_rot'+str(rotulated)+'_'+str(iteration)+'.pkl', 'rb') as f:
    with open('/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/'+output_dataset+'/'+dataset_name+'/new_masks_and_y_'+str(sample_size)+'/mask_rot'+str(rotulated)+'_'+str(iteration)+'.pkl', 'rb') as f:
        aux = pickle.load(f)
        train_mask, val_mask, test_mask = aux[0],aux[1],aux[2]

    # Defining masks
    data['document'].train_mask = train_mask
    data['document'].val_mask = val_mask
    data['document'].test_mask = test_mask

    # Generate MODEL and Optmizer
    if GCN_or_GAT == "GCN":
        model = GNN(hidden_channels=64, out_channels=class_number)
    elif GCN_or_GAT =="GAT":
        model = GAT(hidden_channels=64, out_channels=class_number)

    model = to_hetero(model, data.metadata(), aggr='sum')
    model.to(device) # GPU

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    #contagem de tempo
    start = time.time()

    # Training
    L_loss = []
    for epoch in range(1, 1000):
        loss = train(model,optimizer)
        L_loss.append(loss)
        train_acc, test_acc = test(model, real_y_values)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')
        df_temp = pd.DataFrame(L_loss)
        if len(df_temp >=5):
          loss_es =  pd.DataFrame(L_loss).tail(5)[0].std()
          print('Early stopping: ',loss_es)
          if loss_es <= 0.01:
            break

    #contagem de tempo
    end = time.time()

    # Creating Classification Report
    y_pred, y_true = get_y(model, real_y_values)
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()

    class_dit = classification_report(y_true, y_pred, target_names=target_names, output_dict = True)

    print('\nClassification Report:\n')
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Updating Dataframes
    df_list[i][0].at[model_name+"_"+GCN_or_GAT, str(rotulated)] = class_dit['accuracy']               # acuracia
    df_list[i][1].at[model_name+"_"+GCN_or_GAT, str(rotulated)] = class_dit['macro avg']['precision'] # precision
    df_list[i][2].at[model_name+"_"+GCN_or_GAT, str(rotulated)] = class_dit['macro avg']['recall']    # recall
    df_list[i][3].at[model_name+"_"+GCN_or_GAT, str(rotulated)] = class_dit['macro avg']['f1-score']  # f1-score
    df_list[i][4].at[model_name+"_"+GCN_or_GAT, str(rotulated)] = class_dit['macro avg']['support']   # support
    df_list[i][5].at[model_name+"_"+GCN_or_GAT, str(rotulated)] = (end - start)                       # time


# df_list = []
# for i in range(10):
#     aux_list = []
#     aux_list.append(pd.read_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(i)+"/df_acc.pkl")) # acuracia
#     aux_list.append(pd.read_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(i)+"/df_pre.pkl")) # precision
#     aux_list.append(pd.read_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(i)+"/df_rec.pkl")) # recall
#     aux_list.append(pd.read_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(i)+"/df_f1.pkl"))  # f1-score
#     aux_list.append(pd.read_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(i)+"/df_sup.pkl")) # support
#     aux_list.append(pd.read_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(i)+"/df_time.pkl")) # time
#     df_list.append(aux_list)

for dataset_name in datasets:
    for output_dataset in output_datasets:
        sample_size = output_datasets_dict[output_dataset]

        df_list = []
        for i in range(10):
            aux_list = []
            aux_list.append(pd.read_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(i)+"/df_acc.pkl")) # acuracia
            aux_list.append(pd.read_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(i)+"/df_pre.pkl")) # precision
            aux_list.append(pd.read_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(i)+"/df_rec.pkl")) # recall
            aux_list.append(pd.read_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(i)+"/df_f1.pkl"))  # f1-score
            aux_list.append(pd.read_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(i)+"/df_sup.pkl")) # support
            aux_list.append(pd.read_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(i)+"/df_time.pkl")) # time
            df_list.append(aux_list)

        for keyphrase in keyphrases:
            for rotulated in n_rotulated:
                for iteration in iterations:
                    model_name = dataset_name+"_"+keyphrase
                    print(rotulated,":",model_name,"iteration", iteration)
                    # Define grafo
                    data, real_y_values, class_number, classes = set_up_graph(model_name,rotulated,iteration)

                    # Treina GCN
                    print(f"================ Training GCN ================")
                    training_iteration(iteration, rotulated, 'GCN', data, real_y_values, class_number, classes)

                    # Treina GAT
                    print(f"================ Training GAT ================")
                    training_iteration(iteration, rotulated, 'GAT', data, real_y_values, class_number, classes)
                    
                    print(df_list[iteration][0])

                    df_list[iteration][0]= df_list[iteration][0][["LLM_only","1","5","10","20","30"]]
                    df_list[iteration][1]= df_list[iteration][1][["LLM_only","1","5","10","20","30"]]
                    df_list[iteration][2]= df_list[iteration][2][["LLM_only","1","5","10","20","30"]]
                    df_list[iteration][3]= df_list[iteration][3][["LLM_only","1","5","10","20","30"]]
                    df_list[iteration][4]= df_list[iteration][4][["LLM_only","1","5","10","20","30"]]
                    df_list[iteration][5]= df_list[iteration][5][["LLM_only","1","5","10","20","30"]]

                    df_list[iteration][0].to_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(iteration)+"/df_acc.pkl") # acuracia
                    df_list[iteration][1].to_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(iteration)+"/df_pre.pkl") # precision
                    df_list[iteration][2].to_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(iteration)+"/df_rec.pkl") # recall
                    df_list[iteration][3].to_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(iteration)+"/df_f1.pkl")  # f1-score
                    df_list[iteration][4].to_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(iteration)+"/df_sup.pkl") # support
                    df_list[iteration][5].to_pickle("/home/vitor_vasconcelos/workdir/mestrado_v3/so_LLM/dataframes_"+str(sample_size)+"/"+str(iteration)+"/df_time.pkl") # support

# # Change here to change wich keypharse to use
# keyphrase = "keyphrase2"

# dataset_name = "Dmoz_Computers"
# model_name = dataset_name+"_"+keyphrase
# rotulated=1

# for iteration in range(10):
#     # Define grafo
#     data, real_y_values, class_number, classes = set_up_graph(model_name,rotulated,iteration)

#     # Treina GCN
#     print(f"================ Training GCN ================")
#     training_iteration(iteration, rotulated, 'GCN', data, real_y_values, class_number, classes)

#     # Treina GAT
#     print(f"================ Training GAT ================")
#     training_iteration(iteration, rotulated, 'GAT', data, real_y_values, class_number, classes)