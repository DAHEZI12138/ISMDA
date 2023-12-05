import torch
import torch.nn as nn
from trainer.training_evaluation import model_evaluate, val_self_training

from models.models import EEGNet_ATTEN, Classifier
from dataloader.BCIIV import *

import loss
from torch.autograd import Variable
import time
from pseudo_labeling_util import pseudo_labeling
from utils import parameter_count

def cross_domain_train(src_id, trg_id, src_train_dl, trg_train_dl, Z,
                       device, logger, configs, args):

    # Common Variable Settings
    save_dir = args.save_dir
    save_dir_models = args.save_dir_models
    momentum = configs.momentum
    class_num = configs.num_classes

    # set base network
    param_config = configs.ISMDA_params
    EEGNet_ATTEN_configs = configs.eegnet_atten
    ATDOC_configs = configs.ATDOC_params
    PACC_configs = configs.PACC_params
    Classifier_configs = configs.classifier

    feature_extractor = EEGNet_ATTEN(EEGNet_ATTEN_configs, EEGNet_ATTEN_configs.dropoutRate).float().to(device)

    classifier_1 = Classifier(Classifier_configs).float().to(device)
    trainable0, non_trainable0 = parameter_count(feature_extractor)
    trainable1, non_trainable1 = parameter_count(classifier_1)

    print("param size = %fMB", (trainable0 + trainable1)/1e6)

    # ATDOC
    class_weight_src = torch.ones(class_num, ).to(device)
    max_len = min(len(src_train_dl), len(trg_train_dl))
    args.max_iter = configs.num_epoch * max_len
    mem_fea = torch.rand(len(trg_train_dl.dataset), configs.target_features_length).to(device)# memory bank
    mem_fea = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True) # storage
    mem_cls = torch.ones(len(trg_train_dl.dataset), class_num).to(device) / class_num  # memory bank




    optimizer_encoder = torch.optim.Adam([
         {'params': feature_extractor.parameters()},
         {'params': classifier_1.parameters(), 'lr': configs.lr_c}
     ], configs.lr, betas=(configs.beta1, configs.beta2),weight_decay=configs.weight_decay)

    best_acc = 0
    # the self-training process has two stages
    for round_idx in range(param_config.self_training_iterations):
        # the weights of the two classifiers
        if round_idx == 0:
            src_clf_wt = param_config.src_clf_wt
            mec_cls_wt = param_config.mec_cls_wt
            trg_clf_wt = 0
            time0 = 0
            maxepoch = configs.num_epoch
        else:
            src_clf_wt = param_config.src_clf_wt * 0.1
            mec_cls_wt = param_config.mec_cls_wt * 0
            trg_clf_wt = param_config.trg_clf_wt
            maxepoch = PACC_configs.num_epoch_finetune


        # generate pseudo labels
        val_self_training((feature_extractor, classifier_1), trg_train_dl, device, src_id, trg_id, round_idx, args)

        file_name = f"pseudo_train_{src_id}_to_{trg_id}_round_{round_idx}.pt"
        pseudo_trg_train_dataset = torch.load(os.path.join(args.home_path, "pseudo_label1", file_name))

        # Loading datasets
        pseudo_trg_train_dataset = Load_Dataset_pseudo(pseudo_trg_train_dataset)

        # Dataloader for target pseudo labels
        pseudo_trg_train_dl = torch.utils.data.DataLoader(dataset=pseudo_trg_train_dataset,
                                                          batch_size=configs.batch_size,
                                                          shuffle=True, drop_last=True,
                                                          num_workers=0)

        if round_idx >= 1:
            pseudo_labeling(args, pseudo_trg_train_dl, round_idx, save_dir, save_dir_models, Z, round_idx)

            file_name = f"pseudo_train_{Z}_to_{Z}_round_{round_idx}.pt"
            pseudo_trg_train_dataset = torch.load(os.path.join(args.home_path, "pseudo_label2", file_name))

            # Loading datasets
            pseudo_trg_train_dataset = Load_Dataset_pseudo_2(pseudo_trg_train_dataset)

            # Dataloader for target pseudo labels
            pseudo_trg_train_dl = torch.utils.data.DataLoader(dataset=pseudo_trg_train_dataset,
                                                              batch_size=configs.batch_size,
                                                              shuffle=True, drop_last=False,
                                                              num_workers=0)


        for epoch in range(1, maxepoch + 1):
            t1 = time.time()
            target_loader_iter = iter(pseudo_trg_train_dl)


            loss_list = []
            for i, (src_data, src_labels) in enumerate(src_train_dl):
                time0 = time0 + 1
                feature_extractor.train()
                classifier_1.train()
                src_data = src_data.type(torch.FloatTensor)
                src_labels = src_labels.type(torch.LongTensor)
                src_data = Variable(src_data).to(device)
                src_labels = Variable(src_labels).to(device)

                # pass data through the source model network.
                src_feat = feature_extractor(src_data)# features_source
                src_pred = classifier_1(src_feat)# outputs_source

                # smooth-crossentropy
                src_ = loss.CrossEntropyLabelSmooth(reduction='none', num_classes=class_num, epsilon=configs.smooth)(src_pred, src_labels).to(device)
                weight_src = class_weight_src[src_labels].unsqueeze(0)
                classifier_loss = torch.sum(weight_src * src_) / (torch.sum(weight_src).item())

                total_loss = classifier_loss

                if round_idx == 0:
                    eff = time0 / args.max_iter # increasing with epoch
                else:
                    eff = 1

                # Pseudo Labels
                if (i + 1) % 8 == 0:
                    try:
                        trg_data, pseudo_trg_labels, idx, _ = target_loader_iter.__next__()
                        trg_data = trg_data.type(torch.FloatTensor)
                        pseudo_trg_labels = pseudo_trg_labels.type(torch.LongTensor)
                        trg_data = Variable(trg_data).to(device)
                        pseudo_trg_labels = Variable(pseudo_trg_labels).to(device)
                        trg_feat = feature_extractor(trg_data)  # features_target
                        trg_pred = classifier_1(trg_feat)
                        if round_idx >= 1:
                            model_loss = nn.CrossEntropyLoss()(trg_pred, pseudo_trg_labels).to(device)
                        else:
                            model_loss = 0

                        if round_idx == 0:
                            dis = -torch.mm(trg_feat.detach(), mem_fea.t())  # match against memory bank sample
                            for di in range(dis.size(0)):
                                dis[di, idx[di]] = torch.max(dis)  # normalisation
                            _, p1 = torch.sort(dis, dim=1)  # ranking

                            w = torch.zeros(trg_feat.size(0), mem_fea.size(0)).to(device)
                            for wi in range(w.size(0)):
                                    for wj in range(ATDOC_configs.K):  # find the most similar samples
                                        w[wi][p1[wi, wj]] = 1 / ATDOC_configs.K

                            weight_, pred = torch.max(w.mm(mem_cls), 1) # results
                            loss_ = nn.CrossEntropyLoss(reduction='none')(trg_pred, pred) # weighting of categories
                            classifier_loss = torch.sum(weight_ * loss_) / (torch.sum(weight_).item())
                            mechine_clf_loss = ATDOC_configs.tar_par * eff * classifier_loss

                        elif round_idx >= 1:
                            mechine_clf_loss = 0

                    except:
                        model_loss = 0
                        mechine_clf_loss = 0
                        if round_idx >= 1:
                            target_loader_iter = iter(pseudo_trg_train_dl)
                else:
                    model_loss = 0
                    mechine_clf_loss = 0


                total_loss = trg_clf_wt * model_loss + src_clf_wt * total_loss + mec_cls_wt * mechine_clf_loss
                optimizer_encoder.zero_grad()
                total_loss.backward()
                optimizer_encoder.step()

                loss_list.append([total_loss.item()])

                if ((i + 1) % 8 == 0) & (round_idx == 0):
                    # update memory bank
                    feature_extractor.eval()
                    classifier_1.eval()
                    with torch.no_grad():
                        trg_feat = feature_extractor(trg_data)
                        trg_pred = classifier_1(trg_feat)
                        trg_feat = trg_feat / torch.norm(trg_feat, p=2, dim=1, keepdim=True)
                        softmax_out = nn.Softmax(dim=1)(trg_pred)
                        trg_pred = softmax_out ** 2 / ((softmax_out ** 2).sum(dim=0))
                    # exponential mean shift
                    mem_fea[idx] = (1.0 - momentum) * mem_fea[idx] + momentum * trg_feat.clone()
                    mem_cls[idx] = (1.0 - momentum) * mem_cls[idx] + momentum * trg_pred.clone()


            final_loss, = np.mean(loss_list, 0)

            if epoch % 1 == 0:
                target_loss, target_score, _, _ = model_evaluate((feature_extractor, classifier_1), trg_train_dl, device, False)
                if best_acc < target_score:
                    best_acc = target_score
                    torch.save({'cls_net': (feature_extractor.state_dict(), classifier_1.state_dict())},
                               os.path.join(save_dir + save_dir_models, 'best' + str(Z) + '.pkl'))

                t2 = time.time()
                if round_idx == 0:
                    logger.debug(f'[Epoch : {epoch}/{configs.num_epoch}]')
                else:
                    logger.debug(f'[Epoch : {epoch}/{PACC_configs.num_epoch_finetune}]')
                logger.debug(
                    f'{args.da_method}   SLoss  : {final_loss:.4f}\t   TLoss  : {target_loss:.4f}\t | \t{args.da_method} Accuracy  : {target_score:2.4f}  Best  :{best_acc:2.4f}  |  time {t2-t1:.2f}')
                logger.debug(f'-------------------------------------')


    return target_score, best_acc
