# TVMS-Net

This code is for  `Pancreas Segmentation by Two-View Feature Learning and Multi-Scale Supervision`.  The paper is under review.



## Overview



![fig2](https://s2.loli.net/2021/12/05/tqpojlM5vORwsJf.png)



![bar](https://s2.loli.net/2021/12/15/xp4gYkUcwl5EGei.png)



![loss](https://s2.loli.net/2021/12/09/NCVL7IaBxDvih9F.png)





## Data

- NIH pancreas segmentation dataset：
  - https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT
- Medical Segmentation Decathlon：
  - http://medicaldecathlon.com/



## Pre-trained Models on the NIH Dataset

The 82 cases in the NIH dataset are split into 4 folds:

  * **Fold #0**: testing on Cases 01, 02, ..., 20;
  * **Fold #1**: testing on Cases 21, 22, ..., 40;
  * **Fold #2**: testing on Cases 41, 42, ..., 61;
  * **Fold #3**: testing on Cases 62, 63, ..., 82.

|             |                           **Link**                           | **Accuracy** |
| :---------: | :----------------------------------------------------------: | :----------: |
| **Fold #0** | https://pan.baidu.com/s/1fPCI9EGPm9x_tCzborydHg (code: qpdn) |    85.03%    |
| **Fold #1** | https://pan.baidu.com/s/199OoCaSRaGASbj0OalZxFQ (code: 7uja) |    84.95%    |
| **Fold #2** | https://pan.baidu.com/s/1ZoG36R4bD85WQAD9qxCQog  (code:4eqx ) |    84.59%    |
| **Fold #3** | https://pan.baidu.com/s/1CbjQzJyRNWWYtvF-qo-RcA (code: 1hqj) |    86.22%    |
| **Average** |                                                              |    85.19%    |

## Contact

For any paper related questions, please contact yunjie19@mails.jlu.edu.cn

## Citation

If this paper is helpful for your research, please cite our paper.
