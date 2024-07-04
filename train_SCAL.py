import argparse
import torch.nn.functional as F
import torch
from torch import tensor
from baselines.SCAL.networks import model_instantiation
import numpy as np
from baselines.SCAL.utils import load_data, coarsening
from dataset import load_nc_dataset
# import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_train', action='store_true') # control the coarsening training
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--base_model', type=str, default='GCN') # choose base gnn model
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--normalize_features', action='store_true')
    parser.add_argument('--K', type=int, default=10) # for APPNP-like architecture
    parser.add_argument('--alpha', type=float, default=0.1) # for APPNP
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    args = parser.parse_args()
    # path = "params/"
    # if not os.path.isdir(path):
    #     os.mkdir(path)

    # define # runs number of seeds, for reproducibility
    np.random.seed(0)
    seed = np.random.randint(0,1e6,(args.runs,))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_nc_dataset(args.dataset)

    node_feat = dataset.graph['node_feat'].to(device)
    edge_index = dataset.graph['edge_index'].to(device)
    labels = dataset.label.to(device)


################################################################################
# NOTE: for accurate measure the runtime, we separate large part of code for different settings
################################################################################


####################### full training, dismissing coarsening methods #######################
    if args.full_train: 

        args.num_features, args.num_classes = args.num_features, args.num_classes = dataset.num_features, dataset.num_classes
        model = model_instantiation(args).to(device)
        all_acc = []

        for i in seed:

            # NOTE: for consistent data splits, see data_utils.rand_train_test_idx
            torch.manual_seed(i)
            splits = dataset.get_idx_split()
            
            # original data
            train_idx = splits['train'].to(device)
            val_idx = splits['valid'].to(device)
            test_idx = splits['test'].to(device)

            if args.normalize_features:
                node_feat = F.normalize(node_feat, p=1)

            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            best_val_loss = float('inf')
            val_loss_history = []
            for epoch in range(args.epochs):

                model.train()
                optimizer.zero_grad()
                out = model(node_feat, edge_index)
                loss = F.nll_loss(out[train_idx], labels[train_idx])
                loss.backward()
                optimizer.step()

                model.eval()
                pred = model(node_feat, edge_index)
                val_loss = F.nll_loss(pred[val_idx], labels[val_idx]).item()

                if val_loss < best_val_loss and epoch > args.epochs // 2:
                    best_val_loss = val_loss
                    # torch.save(model.state_dict(), path + 'checkpoint-best-acc.pkl')

                val_loss_history.append(val_loss)
                if args.early_stopping > 0 and epoch > args.epochs // 2:
                    tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])
                    if val_loss > tmp.mean().item():
                        break

            # model.load_state_dict(torch.load(path + 'checkpoint-best-acc.pkl'))
            model.eval()
            pred = model(node_feat, edge_index).max(1)[1]
            test_acc = int(pred[test_idx].eq(labels[test_idx]).sum().item()) / int(test_idx.shape[0])
            print(test_acc)
            all_acc.append(test_acc)

        print('ave_acc: {:.4f}'.format(np.mean(all_acc)))
        print('std: {:.4f}'.format(np.std(all_acc)))


####################### coarsening training, involving coarsening methods #######################
    else:

        args.num_features, args.num_classes, candidate, C_list, Gc_list = coarsening(dataset,
                                                                                     1-args.coarsening_ratio,
                                                                                     args.coarsening_method)
        model = model_instantiation(args).to(device)
        all_acc = []

        for i in seed:

            # NOTE: for consistent data splits, see data_utils.rand_train_test_idx
            torch.manual_seed(i)
            splits, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge = load_data(
                dataset, candidate, C_list, Gc_list)
            # coarsened data
            coarsen_features = coarsen_features.to(device)
            coarsen_train_labels = coarsen_train_labels.to(device)
            coarsen_train_mask = coarsen_train_mask.to(device)
            coarsen_val_labels = coarsen_val_labels.to(device)
            coarsen_val_mask = coarsen_val_mask.to(device)
            coarsen_edge = coarsen_edge.to(device)
            
            # original data
            test_idx = splits['test'].to(device)

            if args.normalize_features:
                node_feat = F.normalize(node_feat, p=1)
                coarsen_features = F.normalize(coarsen_features, p=1)

            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            best_val_loss = float('inf')
            val_loss_history = []
            for epoch in range(args.epochs):

                model.train()
                optimizer.zero_grad()
                out = model(coarsen_features, coarsen_edge)
                loss = F.nll_loss(out[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])
                loss.backward()
                optimizer.step()

                model.eval()
                pred = model(coarsen_features, coarsen_edge)
                val_loss = F.nll_loss(pred[coarsen_val_mask], coarsen_val_labels[coarsen_val_mask]).item()

                if val_loss < best_val_loss and epoch > args.epochs // 2:
                    best_val_loss = val_loss
                    # torch.save(model.state_dict(), path + 'checkpoint-best-acc.pkl')

                val_loss_history.append(val_loss)
                if args.early_stopping > 0 and epoch > args.epochs // 2:
                    tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])
                    if val_loss > tmp.mean().item():
                        break

            # model.load_state_dict(torch.load(path + 'checkpoint-best-acc.pkl'))
            model.eval()
            pred = model(node_feat, edge_index).max(1)[1]
            test_acc = int(pred[test_idx].eq(labels[test_idx]).sum().item()) / int(test_idx.shape[0])
            print(test_acc)
            all_acc.append(test_acc)

        print('ave_acc: {:.4f}'.format(np.mean(all_acc)))
        print('std: {:.4f}'.format(np.std(all_acc)))
