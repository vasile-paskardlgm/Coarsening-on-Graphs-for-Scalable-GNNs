import subprocess
import re
import sys
import argparse
import itertools

# NOTE: more variables / metrics can be involved here, e.g., runtime
def get_results(results):

    acc = float(re.findall(r'.*ave_acc: (.*)', results)[0])
    std = float(re.findall(r'.*std: (.*)', results)[0])

    return [acc, std]

if __name__ == '__main__':

    interpreter = sys.executable

    parser = argparse.ArgumentParser()

    # hyperparameters fixed for all training exps
    parser.add_argument('--full_train', action='store_true')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--base_model', type=str, default='GCN')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--normalize_features', action='store_true')
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    args = parser.parse_args()

    base_arguments = f' --dataset {args.dataset} --base_model {args.base_model} --runs {args.runs} --epochs {args.epochs}\
              --early_stopping {args.early_stopping} --coarsening_ratio {args.coarsening_ratio}\
                  --coarsening_method {args.coarsening_method}'
    if args.full_train:
        base_arguments += ' --full_train'
    if args.normalize_features:
        base_arguments += ' --normalize_features' 


    # hyperparameters required grid-search
    param_dict_APPNP = {'--lr': [1e-3,5e-3,1e-2,5e-2,1e-1,5e-1], 
                        '--weight_decay': [1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1,5e-1],
                        '--dropout': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                        '--K': [2,4,6,8,10],
                        '--alpha': [0.2,0.4,0.6,0.8,1]}
    param_dict_GCN = {'--lr': [1e-3,5e-3,1e-2,5e-2,1e-1,5e-1],
                      '--weight_decay': [1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1,5e-1],
                      '--dropout': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
    
    if args.base_model.upper() == 'GCN':
        param_dict = param_dict_GCN
    elif args.base_model.upper() == 'APPNP':
        param_dict = param_dict_APPNP
    else:
        raise ValueError('Invalid model name')
    

####################### grid search begins #######################
    print(f'Grid search process begins...\n\tmodel: {args.base_model}\n\tdataset: {args.dataset}\
          \n\tcorasening: {(not args.full_train) and args.coarsening_method}\n\thyperparam: {list(param_dict.keys())}')

    best_acc = 0
    best_std = 0

    for combination in itertools.product(*list(param_dict.values())):

        # first obtain the total argument
        argument = ''
        for i, text in enumerate(list(param_dict.keys())):
            argument += f' {text} {list(combination)[i]}'

        # now execute the .py
        results = subprocess.run(f'{interpreter} train_SCAL.py{base_arguments}{argument}', capture_output=True, text=True).stdout
        results = get_results(results=results)
        print(f'Current hyperparam: {combination}', f'Current result: {results[0]} +/- {results[1]}')

        # find the optimal hyperparams
        if (results[0] > best_acc) or ((results[0] == best_acc) and (results[1] < best_std)):
            best_acc = results[0]
            best_std = results[1]
            best_param = combination

    print(f'{args.base_model} on {args.dataset} with {args.coarsening_ratio} {args.coarsening_method} coarsening.')
    print(f'Optimal hyperparam: {best_param}', f'Best result: {best_acc} +/- {best_std}')
