# Tree-based-paper
This repository contains codes for the first arxiv version of the paper **On marginal feature attributions of tree-based models**.
There are three folders:

  1) **TreeSHAP_sanity_check**: This is to confirm the computations done in Example 3.1 regrading the path-dependent TreeSHAP.
  2) **models_metrics**: Two triples of model files are provided. Each triple contains a LightGBM, a CatBoost and an XGBoost model all trained on a public dataset for the same task. The Jupyter notebook explained_variance.ipynb in the folder is to verify the explained variance scores for these models as appeared in Table 4 of the paper. 
  3) **Retrieve_splits**: The goal of the notebook Retrieve_splits.ipynb is to take a saved LightGBM, CatBoost or XGBoost model, decomposing it into its constituent trees, and creating a dictionary for each tree containing information such as distinct features appearing in the tree, tree's depth, the regions cut by the tree etc. This procedure is carried out for ensembles trained for our experiments in §4 and results in Table 5.  
