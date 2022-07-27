from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import torch
import numpy as np
from src.parse import *
from utility.log_helper import *
from utility.data_loader import data_split, load_data
from utility.train_helper import set_random_seed
from data.dataset import RecoDataset
from model.CG_KGR import CGKGR


if __name__ == "__main__":
    PATH = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(PATH)
    ROOT = os.path.join(PATH, '/')
    sys.path.append(ROOT)

    #### Add Configuration ####
    args = parse_args_music()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #### Log Information ####
    log_name = create_log_name(args.saved_dir)

    log_config(path=args.saved_dir, name=log_name, level=logging.DEBUG, console_level=logging.DEBUG, console=True)
    logging.info(args)
    logging.info('CG-KGR experiments.')

    #### Data split ####
    dataset_dir = os.path.join(PATH, args.data_dir, args.data_name)
    if os.path.isfile(os.path.join(dataset_dir, 'train_5.txt')):
        print('The dataset has been split')
    else:
        data_split(args)

    #### Create Dataset ####
    file_name = [1, 2, 3, 4, 5]
    auc = []
    f1 = []

    recall_list = [[] for _ in range(6)]
    ndcg_list = [[] for _ in range(6)]

    k_list = []
    for i in file_name:
        train_file = 'train_' + str(i) + '.txt'
        eval_file = 'eval_' + str(i) + '.txt'
        test_file = 'test_' + str(i) + '.txt'

        logging.info(train_file)

        train_data, eval_data, test_data = load_data(args, train_file, eval_file, test_file)

        #### Define Recommender ####
        recommender = CGKGR(args, train_data, eval_data, test_data).to(device)

        #### Set Random Seed ####
        set_random_seed(args.seed)

        #### Training ####
        # for para in recommender.named_parameters():
        #     print(para)
        best_epoch1, best_epoch2, k_list, best_test_auc, best_test_f1, \
        eval_precision_list, eval_recall_list, eval_ndcg_list, \
        test_precision_list, test_recall_list, test_ndcg_list = recommender.train_model()

        logging.info('')
        auc.append(best_test_auc)
        f1.append(best_test_f1)

        logging.info('CTR Evaluation - Best epoch: %d   corresponding Test auc: %.4f    corresponding Test F1: %.4f'
                     % (best_epoch1, best_test_auc, best_test_f1))

        logging.info('Top@k Evaluation')
        for j, k in enumerate(k_list):
            idx = best_epoch2[j]
            recall_list[j].append(test_recall_list[idx - 1][j])
            ndcg_list[j].append(test_ndcg_list[idx - 1][j])
            logging.info('Top@%d:  Best epoch: %d   corresponding Test R: %.4f    corresponding Test NDCG: %.4f'
                         % (k, best_epoch2[j], test_recall_list[idx - 1][j], test_ndcg_list[idx - 1][j]))
        logging.info('--------------------------------')
        logging.info('')

    for i, k in enumerate(k_list):
        logging.info('Top@%d recommendation:   Avg-best-R %.4f | Avg-best-NDCG: %.4f ' %
                     (k, np.mean(recall_list[i]), np.mean(ndcg_list[i])))

    logging.info(' Avg-best-auc %.4f | Avg-best-F1: %.4f | Max-best-Auc: %.4f | Max-best-F1: %.4f' %
                 (np.mean(auc), np.mean(f1), max(auc), max(f1)))
    logging.info('********************************************************************************************')
    logging.info('********************************************************************************************')






