import torch
import os
from dataloader.HGD import data_generator_HG
from trainer.ISMDA import cross_domain_train
from trainer.training_evaluation import cross_domain_test
from datetime import datetime
from config_files.configs import Config as Configs
from utils import fix_randomness, _logger, copy_Files, plot_all_confusion_matrix, ensure_directories_exist, plot_feature_map
import argparse
import xlwt

######## ARGS ######################
parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', default='experiments_logs', type=str,
                    help='Directory used to save experimental results')

parser.add_argument('--save_dir_models', type=str, default='/saved_models', help='models保存路径')

parser.add_argument('--save_dir_features', type=str, default='/saved_features', help='features保存路径')

parser.add_argument('--save_dir_pre_true_label', type=str, default='/pre_true_label', help='root of features')

parser.add_argument('--save_dir_pesudo_label1', type=str, default='pseudo_label1', help='root of pesudo_label1')

parser.add_argument('--save_dir_pesudo_label2', type=str, default='pseudo_label2', help='root of pesudo_label2')

parser.add_argument('--experiment_description', default='tests', type=str,
                    help='Main experiment Description')

parser.add_argument('--run_description', default='Final_test', type=str,
                    help='Each experiment may have multiple runs, with specific setting in each:')

# Domain adaptation method / Dataset / Model
parser.add_argument('--da_method', default='ISMDA', type=str,
                    help='method selection')

# Experiment setting
parser.add_argument('--num_runs', default=1, type=int,
                    help='Number of consecutive run with different seeds')

parser.add_argument('--device', default='cuda:0', type=str,
                    help='cpu or cuda')
home_dir = os.getcwd()
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')

args = parser.parse_args()

###################################

start_time = datetime.now()
device = torch.device(args.device)
da_method = args.da_method
save_dir = args.save_dir
configs = Configs()

ensure_directories_exist(save_dir, save_dir + args.save_dir_models, save_dir + args.save_dir_features,
                        save_dir + args.save_dir_pre_true_label, args.save_dir_pesudo_label1, args.save_dir_pesudo_label2)

seeds = range(10)

def main_train_cd():
    # find out the domains IDs
    x_domains = [("all", "1"), ("all", "2"), ("all", "3"), ("all", "4"), ("all", "5"), ("all", "6"), ("all", "7"), ("all", "8"), ("all", "9"), ("all", "10"), ("all", "11"), ("all", "12"), ("all", "13"), ("all", "14")]
    classes = ["L H", "R H", "Rest", "Tongue"]

    # Logging
    exp_log_dir = os.path.join(save_dir, args.experiment_description, args.run_description)
    os.makedirs(exp_log_dir, exist_ok=True)

    # save a copy of training files:
    copy_Files(exp_log_dir, da_method)
    ACCS = []
    BESTS = []
    wb = xlwt.Workbook()
    ws = wb.add_sheet('test')

    # loop through domains
    for i,j in enumerate(x_domains):

        src_id = j[0]
        trg_id = j[1]
        # specify number of consecutive runs
        for run_id in range(args.num_runs):
            fix_randomness(seeds[run_id])

            # Logging
            log_dir = os.path.join(exp_log_dir, src_id + "_to_" + trg_id + "_run_" + str(run_id))
            os.makedirs(log_dir, exist_ok=True)
            log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
            logger = _logger(log_file_name)
            logger.debug("=" * 45)
            logger.debug(f'Method:  {da_method}')
            logger.debug("=" * 45)
            logger.debug(f'Source: {src_id} ---> Target: {trg_id}')
            logger.debug(f'Run ID: {run_id}')
            logger.debug("=" * 45)

            # Load datasets data dim is #samples, seq_len, channels
            src_train_dl, trg_train_dl = data_generator_HG(configs, i)


            ACC, BEST_ACC, = cross_domain_train(src_id, trg_id, src_train_dl, trg_train_dl, i, device, logger, configs, args)

            ACCS.append(ACC)
            BESTS.append(BEST_ACC)

            # to test the model and generate results ...
            acc = cross_domain_test(trg_train_dl, device, args, args.save_dir, args.save_dir_models, args.save_dir_pre_true_label, i)
            ws.write(0, int(i), acc.item())

    # plot confusion matrix
    plot_all_confusion_matrix(args.save_dir, classes, args)

    # plot feature map
    for n in range(len(x_domains)):
        plot_feature_map(configs, save_dir, n, device, args.save_dir_features, args.save_dir_models)

    wb.save('experiments_logs/saved_models/test.xls')

    ACC_FINAL = sum(ACCS) / len(ACCS)

    BESTS_FINAL = sum(BESTS) / len(BESTS)

    print(f'teacc {ACC_FINAL:2.2f} BEST_acc {BESTS_FINAL:2.2f}')
    logger.debug(f"Running time: {datetime.now() - start_time}")

if __name__ == "__main__":
    main_train_cd()
