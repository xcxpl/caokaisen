
### task1_main.csv

| dataset   | model   | mode   |   test_acc_mean |   test_acc_std |   time_per_epoch_s_mean |   time_per_epoch_s_std |   train_time_s_mean |   train_time_s_std |   peak_gpu_mb_mean |   peak_gpu_mb_std |
|:----------|:--------|:-------|----------------:|---------------:|------------------------:|-----------------------:|--------------------:|-------------------:|-------------------:|------------------:|
| Amazon    | gat     | full   |          0.8033 |         0.0184 |                  0.0382 |                 0.0016 |               3.818 |             0.1638 |             832.81 |            0.0316 |
| Amazon    | gat     | sample |          0.8165 |         0.0173 |                  0.059  |                 0.0005 |               5.898 |             0.0489 |             528.52 |           55.5609 |
| Amazon    | gcn     | full   |          0.7816 |         0.0332 |                  0.0566 |                 0.0157 |               5.656 |             1.5658 |             352.12 |            0.0789 |
| Amazon    | gcn     | sample |          0.7252 |         0.0727 |                  0.0444 |                 0.0013 |               4.442 |             0.1275 |             436.84 |           57.4509 |
| Amazon    | gin     | full   |          0.2407 |         0.0959 |                  0.0629 |                 0.0002 |               6.291 |             0.0197 |            1786.2  |            0      |
| Amazon    | gin     | sample |          0.682  |         0.0416 |                  0.0424 |                 0.0006 |               4.244 |             0.0626 |            1987.2  |           58.3121 |
| Amazon    | sage    | full   |          0.7067 |         0.0809 |                  0.0593 |                 0.001  |               5.934 |             0.1008 |            1730.7  |            0      |
| Amazon    | sage    | sample |          0.7814 |         0.0224 |                  0.04   |                 0.0007 |               3.999 |             0.0677 |            1901.67 |           52.5309 |
| Citeseer  | gat     | full   |          0.652  |       nan      |                  0.02   |               nan      |               3.99  |           nan      |             206    |          nan      |
| Citeseer  | gat     | sample |          0.647  |       nan      |                  0.0292 |               nan      |               5.84  |           nan      |             226.1  |          nan      |
| Citeseer  | gcn     | full   |          0.643  |       nan      |                  0.008  |               nan      |               1.6   |           nan      |             107.8  |          nan      |
| Citeseer  | gcn     | sample |          0.672  |       nan      |                  0.0196 |               nan      |               3.93  |           nan      |              99.9  |          nan      |
| Citeseer  | gin     | full   |          0.616  |       nan      |                  0.0127 |               nan      |               2.54  |           nan      |             349.3  |          nan      |
| Citeseer  | gin     | sample |          0.623  |       nan      |                  0.018  |               nan      |               3.61  |           nan      |             299.2  |          nan      |
| Citeseer  | sage    | full   |          0.656  |       nan      |                  0.011  |               nan      |               2.2   |           nan      |             355.7  |          nan      |
| Citeseer  | sage    | sample |          0.664  |       nan      |                  0.0168 |               nan      |               3.35  |           nan      |             316.1  |          nan      |
| Cora      | gat     | full   |          0.744  |       nan      |                  0.0151 |               nan      |               3.03  |           nan      |             167.3  |          nan      |
| Cora      | gat     | sample |          0.777  |       nan      |                  0.0393 |               nan      |               7.86  |           nan      |             109    |          nan      |
| Cora      | gcn     | full   |          0.771  |       nan      |                  0.0134 |               nan      |               2.68  |           nan      |              43.4  |          nan      |
| Cora      | gcn     | sample |          0.796  |       nan      |                  0.0318 |               nan      |               6.36  |           nan      |              43.9  |          nan      |
| Cora      | gin     | full   |          0.772  |       nan      |                  0.008  |               nan      |               1.59  |           nan      |             120.7  |          nan      |
| Cora      | gin     | sample |          0.774  |       nan      |                  0.0239 |               nan      |               4.79  |           nan      |             107.2  |          nan      |
| Cora      | sage    | full   |          0.761  |       nan      |                  0.0073 |               nan      |               1.46  |           nan      |             136.9  |          nan      |
| Cora      | sage    | sample |          0.775  |       nan      |                  0.0224 |               nan      |               4.47  |           nan      |             123.4  |          nan      |

### task1_lr.csv

| dataset   | model   |    lr |   test_acc_mean |   test_acc_std |
|:----------|:--------|------:|----------------:|---------------:|
| Cora      | gat     | 0.001 |           0.765 |            nan |
| Cora      | gat     | 0.01  |           0.764 |            nan |
| Cora      | gat     | 0.1   |           0.775 |            nan |
| Cora      | gcn     | 0.001 |           0.767 |            nan |
| Cora      | gcn     | 0.01  |           0.771 |            nan |
| Cora      | gcn     | 0.1   |           0.768 |            nan |
| Cora      | gin     | 0.001 |           0.709 |            nan |
| Cora      | gin     | 0.01  |           0.772 |            nan |
| Cora      | gin     | 0.1   |           0.49  |            nan |
| Cora      | sage    | 0.001 |           0.764 |            nan |
| Cora      | sage    | 0.01  |           0.761 |            nan |
| Cora      | sage    | 0.1   |           0.743 |            nan |

### task1_layers.csv

| dataset   | model   |   num_layers |   test_acc_mean |   test_acc_std |
|:----------|:--------|-------------:|----------------:|---------------:|
| Cora      | gat     |            1 |           0.756 |            nan |
| Cora      | gat     |            2 |           0.763 |            nan |
| Cora      | gat     |            3 |           0.759 |            nan |
| Cora      | gcn     |            1 |           0.769 |            nan |
| Cora      | gcn     |            2 |           0.771 |            nan |
| Cora      | gcn     |            3 |           0.723 |            nan |
| Cora      | gin     |            1 |           0.751 |            nan |
| Cora      | gin     |            2 |           0.772 |            nan |
| Cora      | gin     |            3 |           0.747 |            nan |
| Cora      | sage    |            1 |           0.728 |            nan |
| Cora      | sage    |            2 |           0.761 |            nan |
| Cora      | sage    |            3 |           0.777 |            nan |

### task1_sampler.csv

|   batch_size |   num_neighbors |   test_acc_mean |   test_acc_std |   time_per_epoch_s_mean |   time_per_epoch_s_std |
|-------------:|----------------:|----------------:|---------------:|------------------------:|-----------------------:|
|           16 |               5 |           0.773 |            nan |                  0.085  |                    nan |
|           16 |              10 |           0.774 |            nan |                  0.0787 |                    nan |
|           16 |              20 |           0.769 |            nan |                  0.0822 |                    nan |
|           32 |               5 |           0.753 |            nan |                  0.0448 |                    nan |
|           32 |              10 |           0.777 |            nan |                  0.0467 |                    nan |
|           32 |              20 |           0.789 |            nan |                  0.0496 |                    nan |
|           64 |               5 |           0.782 |            nan |                  0.0284 |                    nan |
|           64 |              10 |           0.799 |            nan |                  0.029  |                    nan |
|           64 |              20 |           0.765 |            nan |                  0.0274 |                    nan |

### task2_main.csv

| dataset   | model   | mode   |   test_auc_mean |   test_auc_std |   test_ap_mean |   test_ap_std |   time_per_epoch_s_mean |   time_per_epoch_s_std |   train_time_s_mean |   train_time_s_std |   peak_gpu_mb_mean |   peak_gpu_mb_std |
|:----------|:--------|:-------|----------------:|---------------:|---------------:|--------------:|------------------------:|-----------------------:|--------------------:|-------------------:|-------------------:|------------------:|
| Amazon    | gat     | full   |          0.9545 |         0.0053 |         0.9443 |        0.0079 |                  0.0408 |                 0.0003 |               4.078 |             0.0297 |            840.7   |            0      |
| Amazon    | gat     | sample |          0.9481 |         0.0018 |         0.9356 |        0.0027 |                  1.255  |                 0.0133 |             125.5   |             1.3253 |            399.033 |            0.7234 |
| Amazon    | gcn     | full   |          0.9664 |         0.0019 |         0.9633 |        0.0021 |                  0.0376 |                 0.0002 |               3.764 |             0.0232 |            802     |            0      |
| Amazon    | gcn     | sample |          0.9629 |         0.0016 |         0.9593 |        0.0021 |                  1.176  |                 0.01   |             117.603 |             1.0023 |            398.8   |            0.7    |
| Amazon    | gin     | full   |          0.7641 |         0.0323 |         0.7434 |        0.0513 |                  0.0614 |                 0      |               6.139 |             0.0032 |           1508.4   |            0      |
| Amazon    | gin     | sample |          0.6756 |         0.0906 |         0.621  |        0.0802 |                  1.125  |                 0.0075 |             112.497 |             0.7508 |           1490.5   |            0.1732 |
| Amazon    | sage    | full   |          0.9612 |         0.0021 |         0.9548 |        0.0026 |                  0.0599 |                 0.0002 |               5.986 |             0.0227 |           1549.6   |            0      |
| Amazon    | sage    | sample |          0.9638 |         0.001  |         0.9575 |        0.0015 |                  1.0945 |                 0.03   |             109.45  |             3.0025 |           1531.7   |            0.1    |
| Citeseer  | gat     | full   |          0.8486 |       nan      |         0.8604 |      nan      |                  0.0197 |               nan      |               1.97  |           nan      |            324.2   |          nan      |
| Citeseer  | gat     | sample |          0.856  |       nan      |         0.8667 |      nan      |                  0.1679 |               nan      |              16.79  |           nan      |            347.4   |          nan      |
| Citeseer  | gcn     | full   |          0.8666 |       nan      |         0.887  |      nan      |                  0.0081 |               nan      |               0.81  |           nan      |            172.1   |          nan      |
| Citeseer  | gcn     | sample |          0.8602 |       nan      |         0.8906 |      nan      |                  0.0943 |               nan      |               9.43  |           nan      |            240.3   |          nan      |
| Citeseer  | gin     | full   |          0.7949 |       nan      |         0.8152 |      nan      |                  0.0109 |               nan      |               1.09  |           nan      |            467.4   |          nan      |
| Citeseer  | gin     | sample |          0.8225 |       nan      |         0.8263 |      nan      |                  0.095  |               nan      |               9.5   |           nan      |            500.7   |          nan      |
| Citeseer  | sage    | full   |          0.8168 |       nan      |         0.8218 |      nan      |                  0.0093 |               nan      |               0.93  |           nan      |            470.7   |          nan      |
| Citeseer  | sage    | sample |          0.8468 |       nan      |         0.8594 |      nan      |                  0.0837 |               nan      |               8.37  |           nan      |            504.3   |          nan      |
| Cora      | gat     | full   |          0.8052 |       nan      |         0.8129 |      nan      |                  0.0163 |               nan      |               1.63  |           nan      |            170     |          nan      |
| Cora      | gat     | sample |          0.815  |       nan      |         0.8242 |      nan      |                  0.1531 |               nan      |              15.31  |           nan      |            173.6   |          nan      |
| Cora      | gcn     | full   |          0.8188 |       nan      |         0.8367 |      nan      |                  0.0191 |               nan      |               1.91  |           nan      |             75.4   |          nan      |
| Cora      | gcn     | sample |          0.8203 |       nan      |         0.8331 |      nan      |                  0.1015 |               nan      |              10.15  |           nan      |             92.2   |          nan      |
| Cora      | gin     | full   |          0.6838 |       nan      |         0.7186 |      nan      |                  0.0077 |               nan      |               0.77  |           nan      |            130.5   |          nan      |
| Cora      | gin     | sample |          0.7691 |       nan      |         0.7584 |      nan      |                  0.0929 |               nan      |               9.29  |           nan      |            144.6   |          nan      |
| Cora      | sage    | full   |          0.7913 |       nan      |         0.785  |      nan      |                  0.0071 |               nan      |               0.71  |           nan      |            146.6   |          nan      |
| Cora      | sage    | sample |          0.8348 |       nan      |         0.8387 |      nan      |                  0.0878 |               nan      |               8.78  |           nan      |            160.8   |          nan      |

### task2_lr.csv

| dataset   | model   |    lr |   test_auc_mean |   test_auc_std |
|:----------|:--------|------:|----------------:|---------------:|
| Cora      | gat     | 0.001 |          0.824  |            nan |
| Cora      | gat     | 0.01  |          0.8056 |            nan |
| Cora      | gat     | 0.1   |          0.5152 |            nan |
| Cora      | gcn     | 0.001 |          0.8085 |            nan |
| Cora      | gcn     | 0.01  |          0.8205 |            nan |
| Cora      | gcn     | 0.1   |          0.6985 |            nan |
| Cora      | gin     | 0.001 |          0.812  |            nan |
| Cora      | gin     | 0.01  |          0.6838 |            nan |
| Cora      | gin     | 0.1   |          0.6102 |            nan |
| Cora      | sage    | 0.001 |          0.8006 |            nan |
| Cora      | sage    | 0.01  |          0.7922 |            nan |
| Cora      | sage    | 0.1   |          0.5016 |            nan |

### task2_layers.csv

| dataset   | model   |   num_layers |   test_auc_mean |   test_auc_std |
|:----------|:--------|-------------:|----------------:|---------------:|
| Cora      | gat     |            1 |          0.6564 |            nan |
| Cora      | gat     |            2 |          0.8001 |            nan |
| Cora      | gat     |            3 |          0.8046 |            nan |
| Cora      | gcn     |            1 |          0.7134 |            nan |
| Cora      | gcn     |            2 |          0.8187 |            nan |
| Cora      | gcn     |            3 |          0.8292 |            nan |
| Cora      | gin     |            1 |          0.7853 |            nan |
| Cora      | gin     |            2 |          0.6838 |            nan |
| Cora      | gin     |            3 |          0.6548 |            nan |
| Cora      | sage    |            1 |          0.7746 |            nan |
| Cora      | sage    |            2 |          0.7907 |            nan |
| Cora      | sage    |            3 |          0.7973 |            nan |

### task2_sampler.csv

|   batch_size |   num_neighbors |   test_auc_mean |   test_auc_std |   time_per_epoch_s_mean |   time_per_epoch_s_std |
|-------------:|----------------:|----------------:|---------------:|------------------------:|-----------------------:|
|          256 |               5 |          0.863  |            nan |                  0.4178 |                    nan |
|          256 |              10 |          0.8534 |            nan |                  0.4302 |                    nan |
|          256 |              20 |          0.8427 |            nan |                  0.4384 |                    nan |
|         1024 |               5 |          0.8307 |            nan |                  0.1219 |                    nan |
|         1024 |              10 |          0.8291 |            nan |                  0.1227 |                    nan |
|         1024 |              20 |          0.834  |            nan |                  0.1236 |                    nan |
|         4096 |               5 |          0.8083 |            nan |                  0.0478 |                    nan |
|         4096 |              10 |          0.8075 |            nan |                  0.0489 |                    nan |
|         4096 |              20 |          0.8263 |            nan |                  0.0483 |                    nan |

### task3_pooling.csv

| dataset     | model   | pool   |   test_acc_mean |   test_acc_std |   test_auc_mean |   test_auc_std |   time_per_epoch_s_mean |   time_per_epoch_s_std |
|:------------|:--------|:-------|----------------:|---------------:|----------------:|---------------:|------------------------:|-----------------------:|
| IMDB-BINARY | gat     | avg    |          0.72   |            nan |        nan      |            nan |                  0.2568 |                    nan |
| IMDB-BINARY | gat     | max    |          0.73   |            nan |        nan      |            nan |                  0.2574 |                    nan |
| IMDB-BINARY | gat     | min    |          0.71   |            nan |        nan      |            nan |                  0.2577 |                    nan |
| IMDB-BINARY | gcn     | avg    |          0.71   |            nan |        nan      |            nan |                  0.1887 |                    nan |
| IMDB-BINARY | gcn     | max    |          0.71   |            nan |        nan      |            nan |                  0.203  |                    nan |
| IMDB-BINARY | gcn     | min    |          0.74   |            nan |        nan      |            nan |                  0.1855 |                    nan |
| IMDB-BINARY | gin     | avg    |          0.72   |            nan |        nan      |            nan |                  0.1702 |                    nan |
| IMDB-BINARY | gin     | max    |          0.76   |            nan |        nan      |            nan |                  0.1688 |                    nan |
| IMDB-BINARY | gin     | min    |          0.76   |            nan |        nan      |            nan |                  0.1691 |                    nan |
| IMDB-BINARY | sage    | avg    |          0.71   |            nan |        nan      |            nan |                  0.1592 |                    nan |
| IMDB-BINARY | sage    | max    |          0.74   |            nan |        nan      |            nan |                  0.1601 |                    nan |
| IMDB-BINARY | sage    | min    |          0.73   |            nan |        nan      |            nan |                  0.158  |                    nan |
| MUTAG       | gat     | avg    |          0.75   |            nan |        nan      |            nan |                  0.0957 |                    nan |
| MUTAG       | gat     | max    |          0.8    |            nan |        nan      |            nan |                  0.093  |                    nan |
| MUTAG       | gat     | min    |          0.8    |            nan |        nan      |            nan |                  0.0944 |                    nan |
| MUTAG       | gcn     | avg    |          0.8    |            nan |        nan      |            nan |                  0.0743 |                    nan |
| MUTAG       | gcn     | max    |          0.8    |            nan |        nan      |            nan |                  0.0823 |                    nan |
| MUTAG       | gcn     | min    |          0.85   |            nan |        nan      |            nan |                  0.0697 |                    nan |
| MUTAG       | gin     | avg    |          0.95   |            nan |        nan      |            nan |                  0.0643 |                    nan |
| MUTAG       | gin     | max    |          0.85   |            nan |        nan      |            nan |                  0.0642 |                    nan |
| MUTAG       | gin     | min    |          0.8    |            nan |        nan      |            nan |                  0.064  |                    nan |
| MUTAG       | sage    | avg    |          0.8    |            nan |        nan      |            nan |                  0.0598 |                    nan |
| MUTAG       | sage    | max    |          0.8    |            nan |        nan      |            nan |                  0.0606 |                    nan |
| MUTAG       | sage    | min    |          0.75   |            nan |        nan      |            nan |                  0.0604 |                    nan |
| PROTEINS    | gat     | avg    |          0.7679 |            nan |        nan      |            nan |                  0.3029 |                    nan |
| PROTEINS    | gat     | max    |          0.7679 |            nan |        nan      |            nan |                  0.2921 |                    nan |
| PROTEINS    | gat     | min    |          0.7857 |            nan |        nan      |            nan |                  0.2932 |                    nan |
| PROTEINS    | gcn     | avg    |          0.7232 |            nan |        nan      |            nan |                  0.2084 |                    nan |
| PROTEINS    | gcn     | max    |          0.7679 |            nan |        nan      |            nan |                  0.2072 |                    nan |
| PROTEINS    | gcn     | min    |          0.7679 |            nan |        nan      |            nan |                  0.2104 |                    nan |
| PROTEINS    | gin     | avg    |          0.7411 |            nan |        nan      |            nan |                  0.2013 |                    nan |
| PROTEINS    | gin     | max    |          0.6964 |            nan |        nan      |            nan |                  0.1955 |                    nan |
| PROTEINS    | gin     | min    |          0.7679 |            nan |        nan      |            nan |                  0.1892 |                    nan |
| PROTEINS    | sage    | avg    |          0.7321 |            nan |        nan      |            nan |                  0.1931 |                    nan |
| PROTEINS    | sage    | max    |          0.6071 |            nan |        nan      |            nan |                  0.1886 |                    nan |
| PROTEINS    | sage    | min    |          0.7589 |            nan |        nan      |            nan |                  0.1938 |                    nan |
| ogbg-molhiv | gat     | avg    |        nan      |            nan |          0.6759 |            nan |                  1.5047 |                    nan |
| ogbg-molhiv | gat     | max    |        nan      |            nan |          0.6607 |            nan |                  1.4891 |                    nan |
| ogbg-molhiv | gat     | min    |        nan      |            nan |          0.6656 |            nan |                  1.4879 |                    nan |
| ogbg-molhiv | gcn     | avg    |        nan      |            nan |          0.6113 |            nan |                  1.1905 |                    nan |
| ogbg-molhiv | gcn     | max    |        nan      |            nan |          0.6082 |            nan |                  1.1424 |                    nan |
| ogbg-molhiv | gcn     | min    |        nan      |            nan |          0.6773 |            nan |                  1.1228 |                    nan |
| ogbg-molhiv | gin     | avg    |        nan      |            nan |          0.6138 |            nan |                  1.0845 |                    nan |
| ogbg-molhiv | gin     | max    |        nan      |            nan |          0.6587 |            nan |                  1.0741 |                    nan |
| ogbg-molhiv | gin     | min    |        nan      |            nan |          0.6557 |            nan |                  1.0722 |                    nan |
| ogbg-molhiv | sage    | avg    |        nan      |            nan |          0.6263 |            nan |                  1.032  |                    nan |
| ogbg-molhiv | sage    | max    |        nan      |            nan |          0.6823 |            nan |                  1.0287 |                    nan |
| ogbg-molhiv | sage    | min    |        nan      |            nan |          0.6809 |            nan |                  1.0185 |                    nan |

### task3_layers.csv

| model   |   num_layers |   test_acc_mean |   test_acc_std |
|:--------|-------------:|----------------:|---------------:|
| gat     |            2 |            0.8  |            nan |
| gat     |            3 |            0.75 |            nan |
| gat     |            4 |            0.85 |            nan |
| gcn     |            2 |            0.75 |            nan |
| gcn     |            3 |            0.8  |            nan |
| gcn     |            4 |            0.85 |            nan |
| gin     |            2 |            0.9  |            nan |
| gin     |            3 |            0.95 |            nan |
| gin     |            4 |            0.95 |            nan |
| sage    |            2 |            0.75 |            nan |
| sage    |            3 |            0.8  |            nan |
| sage    |            4 |            0.8  |            nan |

### task3_lr.csv

| model   |    lr |   test_acc_mean |   test_acc_std |
|:--------|------:|----------------:|---------------:|
| gat     | 0.001 |            0.75 |            nan |
| gat     | 0.01  |            0.8  |            nan |
| gat     | 0.1   |            0.7  |            nan |
| gcn     | 0.001 |            0.7  |            nan |
| gcn     | 0.01  |            0.8  |            nan |
| gcn     | 0.1   |            0.7  |            nan |
| gin     | 0.001 |            0.95 |            nan |
| gin     | 0.01  |            0.9  |            nan |
| gin     | 0.1   |            0.7  |            nan |
| sage    | 0.001 |            0.7  |            nan |
| sage    | 0.01  |            0.8  |            nan |
| sage    | 0.1   |            0.7  |            nan |

### task4_kge.csv

| dataset      | model   |   test_mrr_mean |   test_mrr_std |   test_hits@1_mean |   test_hits@1_std |   test_hits@3_mean |   test_hits@3_std |   test_hits@10_mean |   test_hits@10_std |   time_per_epoch_s_mean |   time_per_epoch_s_std |
|:-------------|:--------|----------------:|---------------:|-------------------:|------------------:|-------------------:|------------------:|--------------------:|-------------------:|------------------------:|-----------------------:|
| countries_S1 | conve   |          0.5083 |            nan |             0.2917 |               nan |             0.5833 |               nan |              1      |                nan |                  0.0668 |                    nan |
| countries_S1 | rotate  |          0.7941 |            nan |             0.625  |               nan |             0.9583 |               nan |              0.9583 |                nan |                  0.0268 |                    nan |
| countries_S1 | transe  |          0.5167 |            nan |             0.25   |               nan |             0.7083 |               nan |              1      |                nan |                  0.0163 |                    nan |
| wn18rr       | conve   |          0.0996 |            nan |             0.0418 |               nan |             0.1238 |               nan |              0.2013 |                nan |                  8.3821 |                    nan |
| wn18rr       | rotate  |          0.3465 |            nan |             0.2808 |               nan |             0.3794 |               nan |              0.4627 |                nan |                  2.1081 |                    nan |
| wn18rr       | transe  |          0.1916 |            nan |             0.0635 |               nan |             0.2731 |               nan |              0.4196 |                nan |                  0.5624 |                    nan |

### task4_dim.csv

| model   |   dim |   test_mrr_mean |   test_mrr_std |   test_hits@10_mean |   test_hits@10_std |
|:--------|------:|----------------:|---------------:|--------------------:|-------------------:|
| rotate  |    50 |          0.6945 |            nan |              0.9167 |                nan |
| rotate  |   100 |          0.7361 |            nan |              1      |                nan |
| rotate  |   200 |          0.6458 |            nan |              1      |                nan |
| transe  |    50 |          0.6681 |            nan |              1      |                nan |
| transe  |   100 |          0.7292 |            nan |              1      |                nan |
| transe  |   200 |          0.9583 |            nan |              1      |                nan |