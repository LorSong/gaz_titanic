# Simply_titanic
Usual titanic eda+modelling (without kaggle submission)  

Has experiments with "double" cross validation and T-test comparison of scores on folds  

Has **report** (on russian) in docs folder  

utils/ - function/modules used in jupiters   

models/cat_boost_7feat_200it.pkl  
Proposed model version, it contains transformer to make simple feature preparation
and CatBoostClassifier trained on 7 features with iteration=200 param selected.   
Model expects **original data** as input, without any preprocessing steps and feature renaming.