import argparse


def parse_args_movie():
    parser = argparse.ArgumentParser(description='Parse for CG-KGR')
    parser = argparse.ArgumentParser(description='Parse for CG-KGR.')
    parser.add_argument('--data_dir', type=str, default='dataset/', help='file path of datasets.')
    parser.add_argument('--data_name', type=str, default='movie', help='select a dataset, e.g., last-fm.')
    parser.add_argument('--kg_file', type=str, default='kg_final.txt', help='select kg file.')
    parser.add_argument('--gpu_id', type=int, default=0, help='select gpu_id')
    parser.add_argument('--node_dim', type=int, default=64, help='the dimension of users, items and entities')
    parser.add_argument('--n_layer', type=int, default=2, help='the number of layers')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size.')
    parser.add_argument('--sample_size', type=int, default=8, help='the size of neighbor samples')
    parser.add_argument('--agg_type', type=str, default='ngh',
                        help='specify the type of aggregation for entities from {sum, concat, ngh}')
    parser.add_argument('--repr_type', type=str, default='combine',
                        help='specify the type of creating user-item representative')
    parser.add_argument('--a', type=float, default=0.8)
    parser.add_argument('--b', type=float, default=0.2)
    parser.add_argument('--n_head', type=int, default=8, help='number of heads in multi-head attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')
    parser.add_argument('--l2_weight', type=float, default=1e-7, help='Lambda when calculating CF l2 loss.')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=10, help='Number of epoch.')
    parser.add_argument('--seed', type=int, default=2021, help='selected seed for training')
    parser.add_argument('--task', type=str, default='ALL', help='[topk, ctr]')

    args = parser.parse_args()
    saved_dir = 'logs/CG-KGR/{}/Dim{}/'.format(args.data_name, args.node_dim)
    args.saved_dir = saved_dir

    return args


def parse_args_book():
    parser = argparse.ArgumentParser(description='Parse for CG-KGR')
    parser.add_argument('--data_dir', type=str, default='dataset/', help='file path of datasets.')
    parser.add_argument('--data_name', type=str, default='book', help='select a dataset, e.g., last-fm.')
    parser.add_argument('--kg_file', type=str, default='kg_final.txt', help='select kg file.')
    parser.add_argument('--gpu_id', type=int, default=0, help='select gpu_id')
    parser.add_argument('--node_dim', type=int, default=64, help='the dimension of users, items and entities')
    parser.add_argument('--n_layer', type=int, default=1, help='the number of layers')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size.')
    parser.add_argument('--sample_size', type=int, default=8, help='the size of neighbor samples')
    parser.add_argument('--agg_type', type=str, default='concat',
                        help='specify the type of aggregation for entities from {sum, concat, ngh}')
    parser.add_argument('--repr_type', type=str, default='mean',
                        help='specify the type of creating user-item representative')
    parser.add_argument('--a', type=float, default=0.5)
    parser.add_argument('--b', type=float, default=0.5)
    parser.add_argument('--n_head', type=int, default=8, help='number of heads in multi-head attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')
    parser.add_argument('--l2_weight', type=float, default=2e-5, help='Lambda when calculating CF l2 loss.')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=10, help='Number of epoch.')
    parser.add_argument('--seed', type=int, default=2021, help='selected seed for training')
    parser.add_argument('--task', type=str, default='ALL', help='[topk, ctr]')

    args = parser.parse_args()
    saved_dir = 'logs/CG-KGR/{}/Dim{}/'.format(args.data_name, args.node_dim)
    args.saved_dir = saved_dir

    return args


def parse_args_dianping():
    parser = argparse.ArgumentParser(description='Parse for CG-KGR')
    parser.add_argument('--data_dir', type=str, default='dataset/', help='file path of datasets.')
    parser.add_argument('--data_name', type=str, default='dianping', help='select a dataset, e.g., last-fm.')
    parser.add_argument('--kg_file', type=str, default='kg_final.txt', help='select kg file.')
    parser.add_argument('--gpu_id', type=int, default=0, help='select gpu_id')
    parser.add_argument('--node_dim', type=int, default=64, help='the dimension of users, items and entities')
    parser.add_argument('--n_layer', type=int, default=1, help='the number of layers')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size.')
    parser.add_argument('--sample_size', type=int, default=8, help='the size of neighbor samples')
    parser.add_argument('--agg_type', type=str, default='concat',
                        help='specify the type of aggregation for entities from {sum, concat, ngh}')
    parser.add_argument('--repr_type', type=str, default='combine',
                        help='specify the type of creating user-item representative ')
    parser.add_argument('--a', type=float, default=0.4)
    parser.add_argument('--b', type=float, default=0.6)
    parser.add_argument('--n_head', type=int, default=8, help='number of heads in multi-head attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')
    parser.add_argument('--l2_weight', type=float, default=1e-7, help='Lambda when calculating CF l2 loss.')
    parser.add_argument('--n_epoch', type=int, default=3, help='Number of epoch.')
    parser.add_argument('--seed', type=int, default=2021, help='selected seed for training')
    parser.add_argument('--task', type=str, default='ALL', help='[topk, ctr]')

    args = parser.parse_args()
    saved_dir = 'logs/CG-KGR/{}/Dim{}/'.format(args.data_name, args.node_dim)
    args.saved_dir = saved_dir

    return args


def parse_args_music():
    parser = argparse.ArgumentParser(description='Parse for CG-KGR')
    # Dataset
    parser.add_argument('--data_dir', type=str, default='dataset/', help='file path of datasets.')
    parser.add_argument('--data_name', type=str, default='music', help='select a dataset, e.g., last-fm.')
    # Knowledge graph
    parser.add_argument('--kg_file', type=str, default='kg_final.txt', help='select kg file.')
    # General hyper-parameters
    parser.add_argument('--gpu_id', type=int, default=0, help='select gpu_id')
    parser.add_argument('--node_dim', type=int, default=16, help='the dimension of users, items and entities')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=10, help='Number of epoch.')
    parser.add_argument('--seed', type=int, default=2021, help='selected seed for training')
    parser.add_argument('--task', type=str, default='ALL', help='[topk, ctr]')
    parser.add_argument('--param_init', type=str, default='xavier_uniform',help='')
    parser.add_argument('--pretrain_flag', type=int, default='0', help='')
    # Hyper-parameters for GCN
    parser.add_argument('--n_layer', type=int, default=1, help='the number of layers')
    parser.add_argument('--sample_size', type=int, default=16, help='the size of neighbor samples')
    parser.add_argument('--agg_type', type=str, default='concat',
                        help='specify the type of aggregation for entities from {sum, concat, ngh}')
    parser.add_argument('--repr_type', type=str, default='max',
                        help='specify the type of creating user-item representative ')
    parser.add_argument('--a', type=float, default=0.5)
    parser.add_argument('--b', type=float, default=0.5)
    parser.add_argument('--n_head', type=int, default=8, help='number of heads in multi-head attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')
    parser.add_argument('--l2_weight', type=float, default=1e-4, help='Lambda when calculating CF l2 loss.')

    args = parser.parse_args()
    saved_dir = 'logs/CG-KGR/{}/Dim{}/'.format(args.data_name, args.node_dim)
    args.saved_dir = saved_dir

    return args
