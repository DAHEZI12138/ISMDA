Codes for ISMDA: EEG-Based Motor Imagery Recognition Framework via Multi-Subject Dynamic Transfer and Iterative Self-Training (DOI: 10.1109/TNNLS.2023.3243339). The performance of the model is evaluated on three publicly available datasets.

## Citation
```
@ARTICLE{10049147,
author={Wang, He and Chen, Peiyin and Zhang, Meng and Zhang, Jianbo and Sun, Xinlin and Li, Mengyu and Yang, Xiong and Gao, Zhongke},
journal={IEEE Transactions on Neural Networks and Learning Systems}, 
title={EEG-Based Motor Imagery Recognition Framework via Multisubject Dynamic Transfer and Iterative Self-Training}, 
year={2023},
volume={},
number={},
pages={1-15},
doi={10.1109/TNNLS.2023.3243339}}
```

## Summary of Results

| Dataset | DRDA | MS-MDA | DAAN | CDAN | MCC | ISMDA |
|-|-|-|-|-|-|-|
| BCI IIV IIa | 52.92 | 55.84 | 62.22 | 63.82 | 66.51 | 69.51 |
| High gamma dataset | - | 77.09 | 77.62 | 78.34 | 81.24 | 82.38 |
| Kwon et al. datasets | - | 57.67 | 80.44 | 89.49 | 85.61 | 90.98 |


## Prepare datasets
We used three public datasets in this study:
- [Dataset IIa of BCI competition IV](https://www.bbci.de/competition/iv/)
- [High gamma dataset](https://github.com/robintibor/high-gamma-dataset)
- [Kwon et al. datasets](http://gigadb.org/dataset/100542)


For the convenience of conducting cross-subject performance testing, we preprocessed the dataset. For example, we merged the train set and test set of the same subject in the High Gamma dataset. Additionally, we followed the settings specified in https://github.com/zhangks98/eeg-adapt/blob/master/preprocess_h5_smt.py to process the Kwon et al. datasets. You can download our processed data at https://www.alipan.com/s/TusMhNbwnpx (9ou7). After downloading, please put it into the DATA folder.


## Training model 
You can update different hyperparameters in the model by updating `config_files/config.py` file.

To train the model, use this command:
```
python train_CD.py --experiment_description XXX --run_description XXX --num_runs 1 --device cuda
```

## Results

The findings encompass the conclusive classification report of the average performance, along with an individualized folder for each cross-domain scenario containing its respective log file and classification report. We also give the confusion matrix for all subjects as well as the feature distribution plots for individual subjects. The optimal model has been provided in the BEST_MODELS folder.

## Contact
He Wang   
School of Electrical and Information Engineering  
Tianjin University, Tianjin, China
Email: hewang@tju.edu.cn
