import argparse
import numpy as np
np.random.seed(0)
from ruamel.yaml import YAML
import os
from models import DMG


def get_args(model_name, dataset, custom_key="", yaml_path=None) -> argparse.Namespace:
    yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    custom_key = custom_key.split("+")[0]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=model_name)
    parser.add_argument("--custom-key", default=custom_key)
    parser.add_argument("--dataset", default=dataset)
    parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
    parser.add_argument('--sparse', type=bool, default=False, help='sparse adjacency matrix')
    parser.add_argument('--iterater', type=int, default=10, help='iterater')
    parser.add_argument('--use_pretrain', type=bool, default=True, help='use_pretrain')
    parser.add_argument('--isBias', type=bool, default=False, help='isBias')
    parser.add_argument('--activation', nargs='?', default='relu')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--gpu_num', type=int, default=4, help='the id of gpu to use')
    parser.add_argument('--seed', type=int, default=0, help='the seed to use')
    parser.add_argument('--test_epo', type=int, default=50, help='test_epo')
    parser.add_argument('--test_lr', type=int, default=0.3, help='test_lr')
    parser.add_argument('--dropout', type=int, default=0.1, help='dropout')
    parser.add_argument('--feature_drop', type=int, default=0.1, help='dropout of features')
    parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
    parser.add_argument("--c_dim", default=8, help="Dimensionality of c", type=int)
    parser.add_argument("--p_dim", default=2, help="Dimensionality of p", type=int)
    parser.add_argument("--lr_max", default=1e0, help="Learning rate for maximization", type=float)
    parser.add_argument("--lr_min", default=1e-3, help="Learning rate for minimization", type=float)
    parser.add_argument("--weight_decay", default=1e-4, help="Weight decay for parameters eta", type=float)
    parser.add_argument("--alpha", default=0.08, help="Reconstruction error coefficient", type=float)
    parser.add_argument("--beta", default=1,  help="Independence constraint coefficient", type=float)
    parser.add_argument("--lammbda", default=1, help="Contrastive constraint coefficient", type=float)
    parser.add_argument("--num_iters", default=200, help="Number of training iterations", type=int)
    parser.add_argument("--inner_epochs", default=10, help="Number of inner epochs", type=int)
    parser.add_argument("--phi_num_layers", default=2, help="Number of layers for phi", type=int)
    parser.add_argument("--phi_hidden_size", default=256, help="Number of hidden neurons for phi", type=int)
    parser.add_argument("--hid_units", default=256, help="Number of hidden neurons", type=int)
    parser.add_argument("--decolayer", default=2, help="Number of decoder layers", type=int)
    parser.add_argument("--neighbor_num", default=300, help="Number of all sampled neighbor", type=int)
    parser.add_argument("--sample_neighbor", default=30, help="Number of sampled neighbor during each iteration", type=int)
    parser.add_argument("--sample_num", default=50, help="Number of sampled edges during each iteration", type=int)
    parser.add_argument("--tau", default=0.5, help="temperature in contrastive loss", type=int)


    with open(yaml_path) as args_file:
        args = parser.parse_args()
        args_key = "-".join([args.model_name, args.dataset, args.custom_key])
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")

    # Update params from .yamls
    args = parser.parse_args()
    return args


def printConfig(args):
    arg2value = {}
    for arg in vars(args):
        arg2value[arg] = getattr(args, arg)
    print(arg2value)


def main():
    args = get_args(
        model_name="DMG",
        dataset="freebase", #acm imdb dblp freebase
        custom_key="Node",  # Node: node classification
    )
    if args.dataset in ["acm", "imdb"]:
        args.num_view = 2
    else:
        args.num_view = 3
    printConfig(args)
    embedder = DMG(args)
    macro_f1s, micro_f1s = embedder.training()
    return macro_f1s, micro_f1s


if __name__ == '__main__':
    macro_f1s, micro_f1s = main()
