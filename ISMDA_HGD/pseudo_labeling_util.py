import random
import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from misc import AverageMeter
from utils import enable_dropout
from config_files.configs import Config as Configs
from models.models import EEGNet_ATTEN, Classifier


def pseudo_labeling(args, data_loader, itr, save_dir, save_dir_models, z, round_idx):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    pseudo_idx = []
    pseudo_target = []
    pseudo_maxstd = []
    gt_target = []
    idx_list = []
    gt_list = []
    target_list = []
    target_data = []
    configs = Configs()
    PACC_configs = configs.PACC_params
    EEGNet_ATTEN_configs = configs.eegnet_atten
    Classifier_configs = configs.classifier

    if not PACC_configs.no_uncertainty:
        f_pass = 30
    else:
        f_pass = 1

    if not PACC_configs.no_progress:
        data_loader = tqdm(data_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets, indexs, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            out_prob = []
            for _ in range(f_pass):
                feature_extractor = EEGNet_ATTEN(EEGNet_ATTEN_configs, round(random.uniform(0.4,0.6),1)).float().to(args.device)
                classifier_1 = Classifier(Classifier_configs).float().to(args.device)
                target_model = (feature_extractor, classifier_1)
                saved_weight = torch.load(os.path.join(save_dir + save_dir_models, 'best' + str(z) + '.pkl'))
                for i, m in enumerate(target_model):
                    m.load_state_dict(saved_weight['cls_net'][i])
                enable_dropout(target_model)

                features = target_model[0](inputs)
                outputs = target_model[1](features)
                out_prob.append(outputs) #for selecting positive pseudo-labels

            out_prob = torch.stack(out_prob)
            out_std = torch.std(out_prob, dim=0)
            out_prob = torch.mean(out_prob, dim=0)
            max_value, max_idx = torch.max(out_prob, dim=1)
            max_std = out_std.gather(1, max_idx.view(-1,1))


            idx_list.extend(indexs.numpy().tolist())
            gt_list.extend(targets.cpu().numpy().tolist())
            target_list.extend(max_idx.cpu().numpy().tolist())

            #selecting positive pseudo-labels
            if not PACC_configs.no_uncertainty:
                selected_idx = (max_value>=PACC_configs.tau_p) * (max_std.squeeze(1) < PACC_configs.kappa_p)
            else:
                selected_idx = max_value>=PACC_configs.tau_p



            pseudo_maxstd.extend(max_std.squeeze(1)[selected_idx].cpu().numpy().tolist())
            pseudo_target.extend(max_idx[selected_idx].cpu().numpy().tolist())
            pseudo_idx.extend(indexs[selected_idx.cpu()].numpy().tolist())
            gt_target.extend(targets[selected_idx].cpu().numpy().tolist())
            target_data.extend(inputs[selected_idx].cpu().numpy().tolist())


            loss = F.cross_entropy(outputs, targets.to(dtype=torch.long))

            losses.update(loss.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not PACC_configs.no_progress:
                data_loader.set_description(
                    "Pseudo-Labeling Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                    batch=batch_idx + 1,
                    iter=len(data_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                ))
        if not PACC_configs.no_progress:
            data_loader.close()


    pseudo_target = np.array(pseudo_target)
    gt_target = np.array(gt_target)
    target_data = np.array(target_data)
    pseudo_maxstd = np.array(pseudo_maxstd)
    pseudo_idx = np.array(pseudo_idx)



    #class balance the selected pseudo-labels
    if itr < PACC_configs.class_blnc-1:
        min_count = 5000000 #arbitary large value
        for class_idx in range(configs.num_classes):
            class_len = len(np.where(pseudo_target==class_idx)[0])
            if class_len < min_count:
                min_count = class_len
        min_count = max(25, min_count) #this 25 is used to avoid degenarate cases when the minimum count for a certain class is very low

        blnc_idx_list = []
        for class_idx in range(configs.num_classes):
            current_class_idx = np.where(pseudo_target==class_idx)
            if len(np.where(pseudo_target==class_idx)[0]) > 0:
                current_class_maxstd = pseudo_maxstd[current_class_idx]
                sorted_maxstd_idx = np.argsort(current_class_maxstd)
                current_class_idx = current_class_idx[0][sorted_maxstd_idx[:min_count]] #select the samples with lowest uncertainty 
                blnc_idx_list.extend(current_class_idx)



        blnc_idx_list = np.array(blnc_idx_list)
        pseudo_target = pseudo_target[blnc_idx_list]
        pseudo_idx = pseudo_idx[blnc_idx_list]
        gt_target = gt_target[blnc_idx_list]
        target_data = target_data[blnc_idx_list]



    pseudo_labeling_acc = (pseudo_target == gt_target)*1
    pseudo_labeling_acc = (sum(pseudo_labeling_acc)/len(pseudo_labeling_acc))*100
    print(f'Pseudo-Labeling Accuracy (positive): {pseudo_labeling_acc}, Total Selected: {len(pseudo_idx)}')
    pseudo_label_dict = {'softmax_labels': pseudo_idx, 'labels':pseudo_target, 'samples': torch.from_numpy(target_data)}
    file_name = f"pseudo_train_{z}_to_{z}_round_{round_idx}.pt"
    torch.save(pseudo_label_dict, os.path.join(args.home_path, args.save_dir_pesudo_label2, file_name))
    print(f'Stage: {round_idx} Complete')