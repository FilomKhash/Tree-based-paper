# Tree-based-paper
This repository contains codes for the second arxiv version of the paper 

<p align=center> <strong>On marginal feature attributions of tree-based models</strong> (https://arxiv.org/abs/2302.08434)

There are four folders:

  1) **TreeSHAP_sanity_check**: This is to confirm the computations done in Example 3.1 regrading the path-dependent TreeSHAP.
  2) **models_metrics**: In §4.1, we experiment with four public datasets. For each of them, a triple consisting of LightGBM, CatBoost and XGBoost models is trained. The model files are provided in the folder along with the Jupyter notebook r2_score.ipynb which replicates their metrics as appeared in Table 5 of the paper. 
  3) **Retrieve_splits**: The goal of the notebook Retrieve_splits.ipynb is to take a saved LightGBM, CatBoost or XGBoost model, decompose it into its constituent trees, and create a dictionary for each decision tree containing information such as distinct features appearing in the tree, tree's depth, the regions cut by the tree etc. This procedure is carried out for the ensembles trained for our experiments in §4.1, and results in Table 6.
  4) **Explainer**: In §4.2, a proprietary implementation of Algorithm 3.12 is used to explain the four CatBoost models previously trained on public datasets. Here, as a sanity check, the [efficiency](https://christophm.github.io/interpretable-ml-book/shapley.html#the-shapley-value-in-detail) property of Shapley values is verified for the outputs of the algorithm.    
