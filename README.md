# Coarsening-on-Graphs-for-Scalable-GNNs
Project about proceeding graph coarsening techniques for scaling up GNN models.

All source code are built as `.py` file format but would be primarily executed on [**Google colab**](https://colab.research.google.com/).

## Key Packages
* torch
* torch-geometric
* ogb
* pygsp
* sortedcontainers
* sklearn
* scipy

## Updates
*Jul 4th, 2024* Baseline [**SCAL**](https://arxiv.org/abs/2106.05150) has been reimplemented into an extensible framework for the follow-up works. 
For the basic execution, you can run `train_SCAL.py` to have a simple trial or run `logger_SCAL.py` to do hyperparameter tuning with grid search.
