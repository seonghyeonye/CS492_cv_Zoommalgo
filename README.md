# SimMixMatch - CS492 CV Project Team Zoommalgo
Semi-supervised learning using NAVER Fashion dataset

## Install Packages
* Apex (using mixed precision for large batch size)
* https://github.com/NVIDIA/apex 

## Path to Best Pretrained Model
* kaist002/fashion_dataset/280 SimMixMatch_e299

## Run Model
* python main.py
* Currently, the hyperparameters are fixed for the best result. If you would like to change hyperparameters, change the values in parser from Options (line 208)
* Trained model will be saved every 50 epoch or when top1 and top5 validation accuracy is updated with the best value.

## Dataset 
* NAVER Fashion dataset consists of 1060 train, 265 validation, and 58735 unlabeled images. In total, labeled data consist of only 1~2%.

## Method
* SimMixMatch is a model combining contrastive learning from the SimCLR model to the MixMatch model. We found that our model has both properties of SimCLR and MixMatch.
* Class SimLoss() in Simloss.py plays an important role in contrastive learning. It calculates loss with cosine similarity and is generalized by K number of augmentations.

## Results
* After fintuning with appropriate hyperparameters, our best validation accuracy score was 20.4% for top1 accuracy and 35.6% for top5 accuracy. We ranked third place in <a href= "https://ai.nsml.navercorp.com/">NSML</a> leaderboard system.


| Model                   | Top1 Validation Accuracy | Top5 Validation Accuracy | Top Test Score |
| -----------------------:| ------------------------:| ------------------------:| --------------:|
| MixMatch Baseline Model | 10.1                     | 18.7                     |                |
| SimMixMatch (Ours)      | **20.4**                 | **35.6**                 | 0.30           |

## References
* <a href = "https://arxiv.org/abs/2002.05709">A Simple Framework for Contrastive Learning of Visual Representations (2020)</a>
* <a href = "https://arxiv.org/abs/1905.02249">MixMatch: A Holistic Approach to Semi-Supervised Learning</a>
* <a href = "https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization">Adaptive Methods for Nonconvex Optimization</a>
