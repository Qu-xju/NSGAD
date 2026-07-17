import fcntl
import types

import numpy as np
import optuna
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from model1 import JacobiConv, Model_low
from utils import *
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import random
import os
import dgl
import pandas as pd
import argparse
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Set argument
# parser = argparse.ArgumentParser(description='CoLA: Self-Supervised Contrastive Learning for Anomaly Detection')
# parser.add_argument('--dataset', type=str, default='cora')  # 'BlogCatalog'  'Flickr'  'ACM'  'cora'  'citeseer'  'pubmed'
# parser.add_argument('--lr', type=float)
# parser.add_argument('--weight_decay', type=float, default=0.0)
# parser.add_argument('--seed', type=int, default=1)
# parser.add_argument('--embedding_dim', type=int, default=64)
# parser.add_argument('--num_epoch', type=int)
# parser.add_argument('--drop_prob', type=float, default=0.0)
# parser.add_argument('--batch_size', type=int, default=300)
# parser.add_argument('--subgraph_size', type=int, default=4)
# parser.add_argument('--readout', type=str, default='avg')  #max min avg  weighted_sum
# parser.add_argument('--auc_test_rounds', type=int, default=256)
# parser.add_argument('--negsamp_ratio', type=int, default=1)
# parser.add_argument('--spec_flag', type=str, default='gcn')
# parser.add_argument('--sub_flag', type=int, default=0)
# parser.add_argument('--adj_normal', type=int, default=0)
# parser.add_argument('--beta', type=float, default=0.)
#
# args = parser.parse_args()

def objective(trial):
    # print(trial)
    Hyparm = {
        # 'expid': 1,
        # 'device': 'cuda:0',
        'dataset': f'{dataset_name}',
        'lr': trial.suggest_categorical('lr1', [0.001,0.005]),
        'seed': trial.suggest_categorical('seed', [1]),
        # 'lr': trial.suggest_categorical('lr', [0.05,0.01, 0.005,0.001,0.0005,0.0001]),
        # 'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1,),
        'weight_decay': trial.suggest_categorical('weight_decay', [0.0, 5e-5, 1e-4, 5e-4, 1e-3]),
        'runs': 1 ,# trial.suggest_int('runs', 1, 10),
        'embedding_dim': trial.suggest_categorical('embedding_dim', [64]),#weibo有修改 其他的都是64
        'patience': trial.suggest_categorical('patience', [40] ),
        'num_epoch': trial.suggest_int('num_epoch', 1,5)*100,
        'drop_prob': trial.suggest_float('drop_prob', 0.0, 0.9,step=0.1),
        'batch_size': trial.suggest_categorical('batch_size', [300]),
        'subgraph_size': trial.suggest_categorical('subgraph_size', [2,4,5,6]),
        'readout': trial.suggest_categorical('readout', ['avg', 'weighted_sum', 'max','min']),
        'auc_test_rounds': 256,#trial.suggest_int('auc_test_rounds', 256),
        'negsamp_ratio': 1,#trial.suggest_int('negsamp_ratio', 1, 5),
        # 'alpha': trial.suggest_categorical('alpha', [1.0]),
        'beta': trial.suggest_float('beta', 0.0, 1.0,step=0.05),
        # 'spec_flag': trial.suggest_categorical('spec_flag', ['bwg']),
        'sub_flag': trial.suggest_categorical('sub_flag', [1]),
        'adj_normal': trial.suggest_categorical('adj_normal', [0, 1]),
        # 'd': trial.suggest_int('d', 1, 6,step=1),
        'blance':trial.suggest_float('blance',0.0,1.0,step=0.1),
        'comment': "_"
    }
    # Hyparm = {
    #     'expid': 1,
    #     'device': 'cuda:0',
    #     'dataset': 'cora',
    #     'lr': 0.001,
    #     'weight_decay': 1e-05,
    #     'runs': 1,
    #     'embedding_dim': 64,
    #     'patience': 40,
    #     'num_epoch': 500,
    #     'drop_prob': 0.3,
    #     'batch_size': 300,
    #     'subgraph_size': 4,
    #     'readout': 'weighted_sum',
    #     'auc_test_rounds': 256,
    #     'negsamp_ratio': 1,
    #     'alpha': 1.0,
    #     'beta': 0.4,
    #     'convfn': 'bwg',
    #     'sub_flag': 1,
    #     'adj_normal': 0,
    #     'd': 2,
    #     'comment':"_"
    # }
    args = types.SimpleNamespace(**Hyparm)
    return train(args)

def train(args):
    if args.lr is None:
        if args.dataset in ['cora','citeseer','pubmed','Flickr']:
            args.lr = 1e-3
        elif args.dataset == 'ACM':
            args.lr = 5e-4
        elif args.dataset == 'BlogCatalog':
            args.lr = 3e-3

    if args.num_epoch is None:
        if args.dataset in ['cora','citeseer','pubmed']:
            args.num_epoch = 100
        elif args.dataset in ['BlogCatalog','Flickr','ACM']:
            args.num_epoch = 400
    batch_size = args.batch_size
    subgraph_size = args.subgraph_size
    blance=args.blance

    print('Dataset: ',args.dataset)


    # Set random seed
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load and preprocess data
    adj, features, labels, idx_train, idx_val,\
    idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)
    raw_features =features.todense()
    features, _ = preprocess_features(features)

    dgl_graph = adj_to_dgl_graph(adj)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]
    if args.adj_normal:
        adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis])

    raw_features = torch.FloatTensor(raw_features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # Initialize model and optimiser
    # if args.spec_flag=='gcn':
    model = Model_low(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
    # if args.spec_flag=='bwg':
        # model = Model_bwg(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout,d=args.d)


    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        features = features.cuda()
        raw_features =raw_features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    if torch.cuda.is_available():
        b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda())
    else:
        b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
    xent = nn.CrossEntropyLoss()
    pdist=nn.PairwiseDistance(p=2)
    mse_loss = nn.MSELoss(reduction='mean')
    cnt_wait = 0
    best = 1e9
    best_t = 0
    batch_num = nb_nodes // batch_size + 1

    added_adj_zero_row = torch.zeros((nb_nodes, 1, subgraph_size))
    added_adj_zero_col = torch.zeros((nb_nodes, subgraph_size + 1, 1))
    added_adj_zero_col[:,-1,:] = 1.
    added_feat_zero_row = torch.zeros((nb_nodes, 1, ft_size))
    if torch.cuda.is_available():
        added_adj_zero_row = added_adj_zero_row.cuda()
        added_adj_zero_col = added_adj_zero_col.cuda()
        added_feat_zero_row = added_feat_zero_row.cuda()

    # Train model
    with tqdm(total=args.num_epoch) as pbar:
        pbar.set_description('Training')
        for epoch in range(args.num_epoch):
            if epoch == 498:
                print("debug#########")

            loss_full_batch = torch.zeros((nb_nodes,1))
            if torch.cuda.is_available():
                loss_full_batch = loss_full_batch.cuda()

            model.train()

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            total_loss = 0.

            # subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
            subv = generate_rw_subgraph(dgl_graph,nb_nodes, subgraph_size)

            subgraphs = [row[1:] + [row[0]] for row in subv]

            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                lbl = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1)

                ba = []
                bf = []
                raw_bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

                if torch.cuda.is_available():
                    lbl = lbl.cuda()
                    added_adj_zero_row = added_adj_zero_row.cuda()
                    added_adj_zero_col = added_adj_zero_col.cuda()
                    added_feat_zero_row = added_feat_zero_row.cuda()

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    raw_cur_feat = raw_features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)
                    raw_bf.append(raw_cur_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]),dim=1)
                raw_bf = torch.cat(raw_bf)
                raw_bf = torch.cat((raw_bf[:, :-1, :], added_feat_zero_row, raw_bf[:, -1:, :]),dim=1)
                logits ,f_1= model(bf, ba,args.sub_flag,blance=blance)
                loss_all = b_xent(logits, lbl)
                loss1 = torch.mean(loss_all)
                loss2 = mse_loss(f_1[:, -2, :], raw_bf[:, -1, :])#重构损失
                # loss = torch.mean(loss_all)
                loss = loss1+args.beta*loss2

                loss.backward()
                optimiser.step()

                loss = loss.detach().cpu().numpy()
                loss_full_batch[idx] = loss_all[: cur_batch_size].detach()

                if not is_final_batch:
                    total_loss += loss

            mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes

            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'best_model_%s.pkl'%(args.dataset))
            else:
                cnt_wait += 1

            pbar.set_postfix(loss=mean_loss)
            pbar.update(1)


    # Test model
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_model_%s.pkl'%(args.dataset)))

    multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
    multi_round_ano_score_p = np.zeros((args.auc_test_rounds, nb_nodes))
    multi_round_ano_score_n = np.zeros((args.auc_test_rounds, nb_nodes))

    with tqdm(total=args.auc_test_rounds) as pbar_test:
        pbar_test.set_description('Testing')
        for round in range(args.auc_test_rounds):

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)

            # subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
            subv = generate_rw_subgraph(dgl_graph,nb_nodes, subgraph_size)

            subgraphs = [row[1:] + [row[0]] for row in subv]

            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                ba = []
                bf = []
                raw_bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

                if torch.cuda.is_available():
                    lbl = lbl.cuda()
                    added_adj_zero_row = added_adj_zero_row.cuda()
                    added_adj_zero_col = added_adj_zero_col.cuda()
                    added_feat_zero_row = added_feat_zero_row.cuda()

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    raw_cur_feat = raw_features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)
                    raw_bf.append(raw_cur_feat)
                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
                raw_bf = torch.cat(raw_bf)
                raw_bf = torch.cat((raw_bf[:, :-1, :], added_feat_zero_row, raw_bf[:, -1:, :]), dim=1)
                with torch.no_grad():
                    logits,f_1 = model(bf, ba,args.sub_flag,blance=blance)
                    dist = pdist(f_1[:, -2, :], raw_bf[:, -1, :])
                    logits = torch.sigmoid(torch.squeeze(logits))
                scaler1=MinMaxScaler()
                scaler2=MinMaxScaler()
                ano_score_1 = - (logits[:cur_batch_size] - logits[cur_batch_size:]).detach().cpu().numpy()
                ano_score_2 = dist.detach().cpu().numpy()
                ano_score_1 = scaler1.fit_transform(ano_score_1.reshape(-1, 1)).reshape(-1)
                ano_score_2 = scaler2.fit_transform(ano_score_2.reshape(-1, 1)).reshape(-1)
                ano_score = ano_score_1 + args.beta * ano_score_2
                # ano_score_n = logits[cur_batch_size:].cpu().numpy()

                multi_round_ano_score[round, idx] = ano_score
                # multi_round_ano_score_p[round, idx] = ano_score_p
                # multi_round_ano_score_n[round, idx] = ano_score_n

            pbar_test.update(1)

    ano_score_final = np.mean(multi_round_ano_score, axis=0)
    # ano_score_final_p = np.mean(multi_round_ano_score_p, axis=0)
    # ano_score_final_n = np.mean(multi_round_ano_score_n, axis=0)
    auc = roc_auc_score(ano_label, ano_score_final)
# #     绘制ROC
    fpr, tpr, thresholds = metrics.roc_curve(ano_label, ano_score_final)
    plot = np.array(list(zip(fpr, tpr)))
    # cur="%s"%( time.strftime('%H:%M:%S', time.localtime()))
    np.savetxt(f'ROCv2_{args.dataset}_NSGAD_ROC.csv', plot, fmt="%f", delimiter=",")
    




    print('AUC:{:.4f}'.format(auc))
    new_data = {
        "dataset": args.dataset,
        "embedding_dim": args.embedding_dim,
        "num_epoch": args.num_epoch,
        "subgraph_size": args.subgraph_size,
        "adj_normalize":args.adj_normal,
        "readout": args.readout,
        # "spec": args.spec_flag,
        "sub": args.sub_flag,
        "auc":auc,
        "beta":args.beta
    }
    pd_tmp = pd.DataFrame([new_data])
    with open("result.csv",'a') as f:

        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (OSError, IOError):
                pass

        try:
            # 读写操作

            try:
                df = pd.read_csv("result.csv")
            except pd.errors.EmptyDataError:
                df = pd.DataFrame(
                    columns=["dataset", "embedding_dim", "num_epoch", "subgraph_size", "adj_normalize", "readout",
                             "alpha",
                             "beta", "d", "convfn", "sub", "auc", "comment"])
            df = pd.concat([df, pd_tmp], ignore_index=True)
            df.to_csv("result.csv", index=False)

        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    print("重构比率",args.beta)
    print(df)
    return auc
if __name__ == '__main__':
    global dataset_name
    dataset_name_s=['cora','Flickr']
    for dataset_name in dataset_name_s:
        study = optuna.create_study(study_name=f"{dataset_name}_parm_v3",direction='maximize',load_if_exists=True,sampler=optuna.samplers.TPESampler(),storage='sqlite:///Res_log_1214.db')
        # study.optimize(objective,n_trials=300)
        print("bestAuc%.4f 对应参数：%s"%(objective(study.best_trial),study.best_params))


