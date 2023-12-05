import os
import torch
import torch.nn as nn
import numpy as np
from models.models import EEGNet_ATTEN, Classifier
from torch.autograd import Variable
from config_files.configs import Config as Configs

def cross_domain_test(tgt_test_dl, device, args, save_dir, save_dir_models, save_dir_pre_true_label, z):
    print('==== Domain Adaptation completed =====')
    print('\n==== Evaluate on test sets ===========')

    configs = Configs()
    EEGNet_ATTEN_configs = configs.eegnet_atten
    Classifier_configs = configs.classifier

    feature_extractor = EEGNet_ATTEN(EEGNet_ATTEN_configs, EEGNet_ATTEN_configs.dropoutRate).float().to(device)

    classifier_1 = Classifier(Classifier_configs).float().to(device)
    target_model = (feature_extractor, classifier_1)
    saved_weight = torch.load(os.path.join(save_dir + save_dir_models, 'best' + str(z) + '.pkl'))

    for i, m in enumerate(target_model):
        m.load_state_dict(saved_weight['cls_net'][i])
    _, acc, pred_labels, true_labels = model_evaluate(target_model, tgt_test_dl, device, False)

    data_save = dict()
    data_save["pred_labels"] = torch.from_numpy(pred_labels)
    data_save["true_labels"] = torch.from_numpy(true_labels)
    file_name = f"final_train_{z}_to_{z}_round_{z}.pt"
    torch.save(data_save, os.path.join(save_dir + save_dir_pre_true_label, file_name))
    print(f'\t{args.da_method} Accuracy     : {acc:2.4f}')
    return acc




def cross_domain_test_fea(src_test_dl, tgt_test_dl, device, save_dir, save_dir_features, z, save_dir_models):

    print('\n==== plot feature map ===========')

    configs = Configs()
    EEGNet_ATTEN_configs = configs.eegnet_atten
    feature_extractor = EEGNet_ATTEN(EEGNet_ATTEN_configs, EEGNet_ATTEN_configs.dropoutRate).float().to(device)
    target_model = feature_extractor
    saved_weight = torch.load(os.path.join(save_dir + save_dir_models, 'best' + str(z) + '.pkl'))
    target_model.load_state_dict(saved_weight['cls_net'][0])
    train_features, train_labels = model_evaluate(target_model, src_test_dl, device, True)
    test_features, test_labels = model_evaluate(target_model, tgt_test_dl, device, True)

    train_feature = []
    test_feature = []
    train_label = []
    test_label = []
    m = 0
    for fea0, lab0 in zip(train_features, train_labels):
        if m == 0:
            train_feature = fea0
            train_label = lab0
        else:
            train_feature = np.concatenate((train_feature, fea0), axis=0)
            train_label = np.concatenate((train_label, lab0), axis=0)
        m = m + 1
    m = 0
    for fea1, lab1 in zip(test_features, test_labels):
        if m == 0:
            test_feature = fea1
            test_label = lab1
        else:
            test_feature = np.concatenate((test_feature, fea1), axis=0)
            test_label = np.concatenate((test_label, lab1), axis=0)
        m = m + 1
    data_save = dict()
    data_save["train_features"] = torch.from_numpy(train_feature)
    data_save["train_labels"] = torch.from_numpy(train_label)
    data_save["test_features"] = torch.from_numpy(test_feature)
    data_save["test_labels"] = torch.from_numpy(test_label)
    file_name = f"final_train_{z}_to_{z}_round_{z}.pt"
    torch.save(data_save, os.path.join(save_dir + save_dir_features, file_name))

def val_self_training(model, valid_dl, device, src_id, trg_id, round_idx, args):
    from sklearn.metrics import accuracy_score
    model[0].eval()
    model[1].eval()

    softmax = nn.Softmax(dim=1)
    all_pseudo_labels = np.array([])
    all_labels = np.array([])
    all_data = []
    all_softmax_labels = []
    with torch.no_grad():
        for data, labels in valid_dl:
            data = data.float().to(device)
            labels = labels.view((-1)).long().to(device)
            features = model[0](data)
            predictions = model[1](features)

            normalized_preds = softmax(predictions)
            pseudo_labels = normalized_preds.max(1, keepdim=True)[1].squeeze()
            all_pseudo_labels = np.append(all_pseudo_labels, pseudo_labels.cpu().numpy())
            all_labels = np.append(all_labels, labels.cpu().numpy())
            all_data.append(data)
            all_softmax_labels.append(normalized_preds)


    all_data = torch.cat(all_data, dim=0)
    all_softmax_labels = torch.cat(all_softmax_labels, dim=0)

    data_save = dict()
    data_save["samples"] = all_data
    data_save["labels"] = torch.LongTensor(torch.from_numpy(all_pseudo_labels).long())
    data_save["softmax_labels"] = all_softmax_labels
    file_name = f"pseudo_train_{src_id}_to_{trg_id}_round_{round_idx}.pt"
    torch.save(data_save, os.path.join(args.home_path, args.save_dir_pesudo_label1, file_name))




def model_evaluate(model, valid_dl, device, features):
    if type(model) == tuple:
        model[0].eval()
        model[1].eval()
    else:
        model.eval()

    if not features:
        total_loss = []
        criterion = nn.CrossEntropyLoss().to(device)
        outs = np.array([])
        trgs = np.array([])

    ps = []
    ys = []

    with torch.no_grad():
        for data, labels in valid_dl:
            data = data.type(torch.FloatTensor)
            labels = labels.type(torch.LongTensor)
            data = Variable(data).to(device)
            labels = Variable(labels).to(device)

            if not features:
                out = model[0](data)
                predictions = model[1](out)

                loss = criterion(predictions, labels)
                total_loss.append(loss.item())

                pred = predictions.max(1, keepdim=False)[1]
                predictions = predictions.argmax(dim=1)
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())
                ps.append(predictions.cpu().detach().numpy())
                ys.append(labels.cpu().numpy())

            else:
                out = model(data)
                ps.append(out.cpu().detach().numpy())
                ys.append(labels.cpu().numpy())

    if not features:
        total_loss = torch.tensor(total_loss).mean()
        ps = np.concatenate(ps)
        ys = np.concatenate(ys)
        acc = np.mean(ys == ps) * 100
        return total_loss, acc, outs, trgs

    else:
        return ps, ys


