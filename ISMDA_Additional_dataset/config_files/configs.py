class Config(object):
    def __init__(self):
        # Cross-domain Training
        self.num_epoch = 300
        self.batch_size = 64

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 1e-3
        self.lr_c = 0.001
        self.weight_decay = 5e-4
        self.smooth = 0.1
        self.momentum = 1.0

        # scheduler
        self.step_size = 10
        self.gamma = 0.1
        self.num_classes = 2
        self.target_features_length = 2880

        # tsne
        self.perplexity = 30

        # model
        self.ISMDA_params = ISMDA_params_configs()
        self.eegnet_atten = EEGNet_ATTEN_configs()
        self.ATDOC_params = ATDOC_params_configs()
        self.PACC_params = PACC_params_configs()
        self.classifier = Classifier_configs()


class ISMDA_params_configs(object):
    def __init__(self):

        self.src_clf_wt = 1
        self.trg_clf_wt = 2
        self.mec_cls_wt = 1
        self.self_training_iterations = 4


class EEGNet_ATTEN_configs(object):
    def __init__(self):

        self.afr_reduced_cnn_size = 48
        self.Chans = 62
        self.dropoutRate = 0.5
        self.kernLength1 = 36
        self.kernLength2 = 24
        self.kernLength3 = 18
        self.F1 = 8
        self.D = 2
        self.expansion = 4

class ATDOC_params_configs(object):
    def __init__(self):
        self.tar_par = 0.02
        self.K = 5

class PACC_params_configs(object):
    def __init__(self):
        # num of finetune
        self.num_epoch_finetune = 100
        # total number of class balanced iterations
        self.class_blnc = 3
        # don't use uncertainty in the pesudo-label selection
        self.no_uncertainty = False
        # don't use progress bar
        self.no_progress = False
        # confidece threshold for positive pseudo-labels
        self.tau_p = 0.7
        # confidece threshold for negative pseudo-labels
        self.tau_n = 0.05
        # uncertainty threshold for positive pseudo-labels
        self.kappa_p = 0.05
        # uncertainty threshold for negative pseudo-labels
        self.kappa_n = 0.005
class Classifier_configs(object):
    def __init__(self):
        # Classifier
        self.num_classes = 2
        self.features_len = 2880
