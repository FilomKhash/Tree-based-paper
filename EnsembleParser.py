############################################# 
# A script for parsing and analyzing tree structures for trained CatBoost, LightGBM or XGBoost ensembles.
# Contains the class Parser.  
############################################# 
# Code author:
# 	Khashayar Filom
# Consultant:
# 	Alexey Miroshnikov
############################################# 
# version 1 (Feb 2024) 
# packages:
#	catboost 1.2
#	lighgbm  4.3
#	xgboost  1.6
##############################################


# Default Python packages
import numpy as np
import pandas as pd
import json
import os
from copy import deepcopy
import pickle
import datetime
import warnings

# Importing boosting libraries
from catboost import CatBoostRegressor, CatBoostClassifier
import lightgbm
from lightgbm import LGBMRegressor, LGBMClassifier
import xgboost
from xgboost import XGBClassifier, XGBRegressor


class Parser:
    '''
    A class for parsing a trained CatBoost, LightGBM or XGBoost ensemble: retrieves the tree structures and returns various statistics.
    
    
    Attributes
    ----------
    library_type: str
        Either 'CatBoost', 'LightGBM' or 'XGBoost'.
    
    ml_model: CatBoostRegressor or CatBoostClassifier or LGBMRegressor or LGBMClassifier or lightgbm.Booster or XGBClassifier or XGBRegressor or xgboost.Booster.
        The loaded/provided ensemble model.
        
    num_model_vars: int
        The number of features that appear in trees of the ensemble.
        
    num_trees: int
        The number of trees in the ensemble.
        
    Loaded: dictionary or dataframe
        A comprehensible dumped version of the ensemble model under consideration capturing the structure of the constituent trees.
        
    feature_names: list
        The list of features on which the model was trained. 
        
    
    Methods
    -------
    parse_tree(tree_index=None):
        Parses a tree of the ensemble (or all trees) as a dictionary which records the depth, the features and the thresholds of the tree's splits, the number of leaves, leaf values and probabilities, and the bounds describing the region corresponding to each leaf. 
        
    tree_average(tree_index=None):
        Returns the average of leaf values of a tree (or all trees).
    
    trees_depths():
        Returns the depths of all trees in the ensemble as a list. 
         
    trees_leaf_nums():
        Returns the number of leaves of all trees in the ensemble as a list.
    
    trees_distinct_features():
        Returns the indices of distinct features on which individual trees from the ensemble split as a list of lists.    
    
    trees_distinct_features_num():
        Returns the number of distinct features on which trees in the ensemble split as a list.    
        
    feature_occurences(num_training_features='auto'):
        Returns a list consisting of the number of trees that split on a feature for any feature of the training data.
        
    print_ensemble_statistics(num_training_features='auto'):
        Prints various statistics about the ensemble including:
            the number of trees in the ensemble;
            the average depth of a tree from the ensemble;
            the average number of leaves of a tree from the ensemble;
            the average number of distinct features per tree in the ensemble.
    '''
    ################################## START INIT ########################################################
    # The trained ensemble model will be dumped as a dataframe or a dictionary. 
    # If a path to a model is provided instead of a model, then the model will be loaded. 
    # The type of the boosting library will be recorded as well as the number of features and trees. 
    
    def __init__(self,model,model_type=None,is_pickle=False):
        if model is None:
            raise ValueError("Either a model or a path to a model should be provided.")
        
        # Case 1) A path to the model file is provided which is not a pickle.
        if isinstance(model,str) and not is_pickle:
            input_path = model
            assert os.path.exists(input_path), f"[Error] The file {input_path} does not exist."
            if model_type not in {'CatBoost','LightGBM','XGBoost'}:
                raise ValueError("The type of the boosting library should be provided as 'CatBoost', 'LightGBM' or 'XGBoost' when the input is a path to a saved model and is_pickle is False.")
            self._type = model_type
             
            # Both CatBoostRegressor and CatBoostClassifier can be loaded as a regressor. The value of the 'features_info' key of the resulting dictionaries may be different. 
            # But the structure of the oblivious trees can be recovered the same no matter if we load as a regressor or a classifier. 
            if self._type == 'CatBoost':
                try:
                    model_cat = CatBoostRegressor()             
                    model_cat.load_model(input_path)
                    self._ml_model = model_cat
                except Exception as e:
                    if type(e).__name__ == 'CatBoostError':
                        raise ValueError(f" The file at the provided path {input_path} cannot be loaded as a CatBoost model.")
                time_stamp = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
                model_cat.save_model('temp'+time_stamp,format='json')
                with open('temp'+time_stamp) as file:
                    dictionary_catboost = json.load(file)
                try:
                    os.remove('temp'+time_stamp)
                except PermissionError:
                    pass
                self._Loaded = dictionary_catboost              # A dictionary describing the CatBoost ensemble. 
                
            if self._type == 'LightGBM':
                try:
                    model_lgbm = lightgbm.Booster(model_file = input_path)
                    self._ml_model = model_lgbm
                except Exception as e:
                    if type(e).__name__ == 'LightGBMError':
                        raise ValueError(f" The file at the provided path {input_path} cannot be loaded as a LightGBM model.")
                dictionary_lgbm = model_lgbm.dump_model()
                self._Loaded = dictionary_lgbm                  # A dictionary describing the LightGBM ensemble. 
                
            if self._type == 'XGBoost':
                try:
                    model_xgb = xgboost.Booster()
                    model_xgb.load_model(input_path)
                    self._ml_model = model_xgb
                except Exception as e:
                    if type(e).__name__ == 'XGBoostError':
                        raise ValueError(f" The file at the provided path {input_path} cannot be loaded as an XGBoost model.")    
                df_xgb = model_xgb.trees_to_dataframe()
                self._Loaded = df_xgb                           # A dataframe describing the XGBoost ensemble. 
                
        # Case 2) An ensemble model is provided, or a path to a pickled model.
        else:
            if is_pickle:
                input_path = model
                assert isinstance(input_path,str), "[Error] When is_pickle is True, model should be a string describing the path to a pickled model."
                assert os.path.exists(input_path), f"[Error] The file {input_path} does not exist."
                try:
                    ml_model = pickle.load(open(input_path,'rb'))
                except Exception as e:
                    if type(e).__name__ == 'UnpicklingError':
                         raise ValueError(f" The file at the provided path {input_path} cannot be unpickled.")  
            else:
                ml_model = model
            if isinstance(ml_model,(CatBoostClassifier,CatBoostRegressor)):
                self._type = 'CatBoost'
                time_stamp = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
                ml_model.save_model('temp'+time_stamp,format='json')
                with open('temp'+time_stamp) as file:
                    dictionary_catboost = json.load(file)
                try:
                    os.remove('temp'+time_stamp)
                except PermissionError:
                    pass
                self._Loaded = dictionary_catboost              # A dictionary describing the CatBoost ensemble.
            elif isinstance(ml_model,(LGBMClassifier,LGBMRegressor)):
                self._type = 'LightGBM'
                dictionary_lgbm = ml_model.booster_.dump_model()                
                self._Loaded = dictionary_lgbm                  # A dictionary describing the LightGBM ensemble.
            elif isinstance(ml_model,(XGBClassifier,XGBRegressor)):
                self._type = 'XGBoost'
                time_stamp = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
                ml_model.save_model('temp'+time_stamp+'.model')
                temp_booster = xgboost.Booster()
                temp_booster.load_model('temp'+time_stamp+'.model')
                df_xgb = temp_booster.trees_to_dataframe()
                try:
                    os.remove('temp'+time_stamp+'.model')
                except PermissionError:
                    pass
                self._Loaded = df_xgb                           # A dataframe describing the XGBoost ensemble. (It was more convenient to save and then load so that the Feature column of this dataframe becomes simpler.)
            else:
                raise ValueError("The provided model should be a CatBoost, LightGBM or XGBoost classifier or regressor.")     
            self._ml_model = ml_model
            
        # Recording the ensemble size and the number of features on which the ensemble splits. 
        # Checking there are no categorical features (because the way the splits w.r.t. them are recorded is usually complicated.)
        if self._type == 'CatBoost':
            if set(self._Loaded['features_info'].keys()) != {'float_features'}:
                raise NotImplementedError("Ensembles trained on categorical features are not supported.")
            self._num_model_vars = len(self._Loaded['features_info']['float_features'])
            self._num_trees = len(self._Loaded['oblivious_trees'])
            
        if self._type == 'LightGBM':
            if len(self._Loaded['pandas_categorical']) != 0:
                raise NotImplementedError("Ensembles trained on categorical features are not supported.")
            self._num_model_vars = len(self._Loaded['feature_infos'])
            self._num_trees = len(self._Loaded['tree_info'])
        
        if self._type == 'XGBoost':
            if not pd.isnull(self._Loaded['Category']).all():
                raise NotImplementedError("Ensembles trained on categorical features are not supported.")
            self._num_model_vars = len(self._Loaded[self._Loaded['Feature'].apply(lambda x:x[0]=='f')]['Feature'].unique())
            self._num_trees = len(self._Loaded['Tree'].unique())
            
        # Recovering the names of features of the training data.
        if self._type == 'CatBoost':
            self._feature_names = self._ml_model.get_feature_importance(prettified = True)['Feature Id'].tolist()
        elif self._type == 'LightGBM':
            self._feature_names = self._Loaded['feature_names']
        else:
            if isinstance(self._ml_model,(XGBClassifier,XGBRegressor)):           # One needs to extract the booster object in this case to check the feature names.
                self._feature_names = self._ml_model.get_booster().feature_names  # May be None if one loads a saved XGBoost model.
            else:                                                                 # self._ml_model is a booster object itself.
                self._feature_names = self._ml_model.feature_names                # May be None if one loads a saved XGBoost model.
    ################################## END INIT ############################################################# 
    
    
    ################################## Properties/Attributes ################################################
    @property
    def library_type(self):
        '''
        Returns the type of the boosting library under consideration as 'CatBoost', 'LightGBM' or 'XGBoost'. 
        '''
        return self._type
    
    @property
    def ml_model(self):
        '''
        Returns the inputted/loaded CatBoost, LightGBM or XGBoost ensemble (it may be a booster object in the case of the latter two libraries.)
        '''
        return deepcopy(self._ml_model)
    
    @property
    def num_model_vars(self):
        '''
        Returns the number of features that appear in trees of the ensemble.
        (Caution: This may be strictly smaller than the number of features of the training data.) 
        '''
        return self._num_model_vars
    
    @property
    def num_trees(self):
        '''
        Returns the number of trees in the ensemble. 
        '''
        return self._num_trees
    
    @property
    def Loaded(self):
        '''
        Returns the ensemble under consideration in a comprehensible form, a dictionary in the case of CatBoost or LightGBM, and a dataframe in the case of XGBoost. 
        '''
        return self._Loaded.copy()
    
    @property
    def feature_names(self):
        '''
        Returns the names of features of the training data. 
        It may be None in case a saved XGBoost model is loaded. 
        # (https://stackoverflow.com/questions/54933804/how-to-restore-the-original-feature-names-in-xgboost-feature-importance-plot-af/65991361#65991361)
        '''
        if self._type == 'XGBoost' and self._feature_names == None:
            warnings.warn("The names of features of the training data cannot be recovered from the loaded XGBoost model.")
            return    
        else:
            return deepcopy(self._feature_names)
	#########################################################################################################
 
 
    ################################## Main Methods #########################################################
    def parse_tree(self,tree_index=None):
        '''
        Parses a particular tree of the ensemble, or all of its trees. 
        
        Input
        -----
        tree_index: int
            It should be a non-negative integer smaller than the total number of trees in the ensemble (self.num_trees).
            
        Output
        ------
        A dictionary info with keys info.keys()=['depth', 'n_leaves', 'splits', 'distinct_feature_indx', 'regions'] which describes the tree at the given index.
            info['depth']: the depth of the tree; info['n_leaves']: the number of leaves; info['splits]: a list of all splits happening in the tree recorded in the form of (feature_index,threshold);
            info['distinct_feature_indx']: records indices of distinct features with respect to which splittings occur in the ascending order;
            info['regions'] is a list of dictionaries, each describing the region determined by a leaf. The keys for each element of info['regions'] are 'value', 'weight', 'probability', and the indices of features which should satisfy non-trivial bounds in the region.    
        When tree_index is None, all trees in the ensemble are parsed and a list of such dictionaries is returned.  
        '''
        assert self._type in {'CatBoost','LightGBM','XGBoost'}
        if tree_index != None:
            if tree_index not in range(self._num_trees):
                raise ValueError(f'The ensemble has {self._num_trees} trees. The provided index {tree_index} must be in the range 0<= <{self._num_trees} because .')
            if self._type == 'CatBoost':
                tree_structure = self._Loaded['oblivious_trees'][tree_index]                           # Isolating the part of the dumped dictionary that captures the desired tree. 
                return self._retrieve_catboost(tree_structure)
            elif self._type == 'LightGBM':
                tree_structure = self._Loaded['tree_info'][tree_index]['tree_structure']               # Isolating the part of the dumped dictionary that captures the desired tree.
                return self._retrieve_lgbm(tree_structure)
            else:
                tree_structure = self._Loaded[self._Loaded['Tree']==tree_index].reset_index(drop=True) # Isolating the part of the dumped dataframe that captures the desired tree.
                return self._retrieve_xgb(tree_structure)      
        else:
            parsed_list = []
            if self._type == 'CatBoost':
                for tree_index in range(self._num_trees):
                    tree_structure = self._Loaded['oblivious_trees'][tree_index]
                    parsed_list += [self._retrieve_catboost(tree_structure)]
                return parsed_list
            elif self._type == 'LightGBM':
                for tree_index in range(self._num_trees):
                    tree_structure = self._Loaded['tree_info'][tree_index]['tree_structure']
                    parsed_list += [self._retrieve_lgbm(tree_structure)]
                return parsed_list
            else: 
                for tree_index in range(self._num_trees):
                    tree_structure = self._Loaded[self._Loaded['Tree']==tree_index].reset_index(drop=True)
                    parsed_list += [self._retrieve_xgb(tree_structure)]
                return parsed_list
            
            
    def tree_average(self,tree_index=None):
        '''
        Returns the average of leaf values of a tree. 
        (For an ensemble classifier, the leaf values are logit probability values, i.e. raw prediction scores.) 
        
        Input
        -----
        tree_index: int
            It should be a non-negative integer smaller than the total number of trees in the ensemble (self.num_trees).
            
        Output
        ------
        The average of leaf values of the tree at the given index.
        When tree_index is None, the average is computed for every tree from the ensemble, and they are returned as a list.  
        '''
        if tree_index != None:
            if tree_index not in range(self._num_trees):
                raise ValueError(f'The ensemble has {self._num_trees} trees. The provided index {tree_index} must be in the range 0<= <{self._num_trees}.')
            info = self.parse_tree(tree_index)
            leaf_probs = []
            leaf_values = []
            for region in info['regions']:
                leaf_probs += [region['probability']]
                leaf_values += [region['value']]
            return np.dot(np.asarray(leaf_probs),np.asarray(leaf_values))
        else:
            info_list = self.parse_tree()
            average_list =[]
            for info in info_list:
                leaf_probs = []
                leaf_values = []
                for region in info['regions']:
                    leaf_probs += [region['probability']]
                    leaf_values += [region['value']]
                average_list += [np.dot(np.asarray(leaf_probs),np.asarray(leaf_values))]
            return average_list
        
    
    def trees_depths(self):
        '''
        Returns the depths of all trees in the ensemble as a list.    
        '''
        info_list = self.parse_tree()
        depth_list =[]
        for info in info_list:
            depth_list += [info['depth']]
        return depth_list
        
        
    def trees_leaf_nums(self):
        '''
        Returns the number of leaves of all trees in the ensemble as a list.    
        '''
        info_list = self.parse_tree()
        leaf_num_list =[]
        for info in info_list:
            leaf_num_list += [info['n_leaves']]
        return leaf_num_list        
    
    
    def trees_distinct_features(self):
        '''
        Returns the indices of distinct features on which individual trees from the ensemble split as a list of lists.    
        '''
        info_list = self.parse_tree()
        distinct_features_lists =[]
        for info in info_list:
            distinct_features_lists += [info['distinct_feature_indx']]
        return distinct_features_lists
    
    
    def trees_distinct_features_num(self):
        '''
        Returns the number of distinct features on which trees in the ensemble split as a list.    
        '''
        info_list = self.parse_tree()
        distinct_features_num_lists =[]
        for info in info_list:
            distinct_features_num_lists += [len(info['distinct_feature_indx'])]
        return distinct_features_num_lists
    
    
    def feature_occurences(self,num_training_features='auto'):
        '''
        Returns a list consisting of the number of trees that split on a feature for any feature of the training data. 
        
        Input
        -----
        num_training_features: int
            The number of features of the training data. If 'auto', the function tries to recover that number from the loaded model if possible.
            (Caution: This may be strictly larger than the number of features on which the trees actually split.)
            (The number inputted by the user will be double checked with the number of training features retrieved from the model if available.)
            
        Output
        ------
        A list whose element at index i is the number of trees of the ensemble which split on feature index i of the training data.  
        '''
        if num_training_features == 'auto':
            if self._feature_names == None:                                                     # The number of training features cannot be recovered from the model.
                raise ValueError(f"The number of features in the training data couldn't be recovered from the {self._type} model, and should be inputted.")
            else:
                num_training_features = len(self._feature_names) 
        if isinstance(num_training_features,int) and num_training_features>0:
            pass
        else:
            raise ValueError('The number of features in the training data should be inputted as a positive integer.')
        if self._feature_names != None and len(self._feature_names) != num_training_features:
            raise ValueError(f'The input {num_training_features} does not match the number features on which the {self._type} model is trained which is {len(self._feature_names)}.')
        assert self.num_model_vars <= num_training_features                                     # A sanity check 
        feature_occurrences=np.zeros(num_training_features)
        info_list = self.parse_tree()
        for info in info_list:
            for i in info['distinct_feature_indx']:
                feature_occurrences[i]+=1
        return feature_occurrences.tolist()
    
    
    def print_ensemble_statistics(self,num_training_features='auto'):
        '''
        Prints various statistics about the ensemble:
            the number of features on which the trees split;
            the number of features of the training data (only if it is provided/can be retrieved);
            the number of trees in the ensemble;
            the average depth of a tree from the ensemble;
            the average number of leaves of a tree from the ensemble;
            the average number of distinct features per tree in the ensemble;
            average number of times a feature from the training data appears in a tree from the ensemble (only if the number of features of the training data is provided/can be retrieved).
            
        Input
        -----
        num_training_features: int or 'auto'
            The number of features of the training data. If 'auto', the function tries to recover that number from the loaded model if possible.
            (Caution: This may be strictly larger than the number of features on which the trees actually split.)
            (The number inputted by the user will be double chekced with the number of training features retrieved from the model if available.)
        '''
        if num_training_features == 'auto':
            if self._feature_names == None:                                                     # The number of training features cannot be recovered from the model.
                num_training_features = None
            else:
                num_training_features = len(self._feature_names)                                # The number of training features can be recovered from the model.
        if num_training_features != None and self._feature_names != None and len(self._feature_names) != num_training_features:
            raise ValueError(f'The input {num_training_features} does not match the number features on which the {self._type} model is trained which is {len(self._feature_names)}.')
        else: 
            if num_training_features != None:
                assert self.num_model_vars <= num_training_features, f"[Error] The number {num_training_features} for the number of training features cannot be smaller than the number of model variables which is {self._num_model_vars}."
            feature_occurrences = np.zeros(num_training_features)
        info_list = self.parse_tree()
        depth_list = []
        leaf_num_list = []
        distinct_features_num_lists = []
        for info in info_list:
            depth_list += [info['depth']]
            leaf_num_list += [info['n_leaves']]
            distinct_features_num_lists += [len(info['distinct_feature_indx'])]
            if num_training_features:
                for i in info['distinct_feature_indx']:
                    feature_occurrences[i] += 1
        
        print(f'The number of features on which the trees split: {self._num_model_vars}.\n')
        if num_training_features:
            print(f'The number of features of the training data: {num_training_features}.\n')
        print(f'The number of trees in the ensemble: {self._num_trees}.\n')
        print(f'The average depth of a tree from the ensemble: {np.average(np.asarray(depth_list))}.\n')
        print(f'The average number of leaves of a tree from the ensemble: {np.average(np.asarray(leaf_num_list))}.\n')
        print(f'The average number of distinct features per tree in the ensemble: {np.average(np.asarray(distinct_features_num_lists))}.\n')
        if num_training_features:
            print(f'The average number of times a feature from the training data appears in a tree from the ensemble: {np.average(np.asarray(feature_occurrences))}.')        
    #########################################################################################################
      
 
    ################################## Parsing a CatBoost Tree ##############################################
    @staticmethod
    def _retrieve_catboost(tree_structure):
        '''
        Parses a tree of a trained CatBoost ensemble.
        
        Input
        -----
        The input tree_structure is a dictionary describing a tree of a trained CatBoost ensemble (with keys 'leaf_values', 'leaf_weights' and 'splits'). 
        
        Output
        ------
        A dictionary info with keys info.keys()=['depth', 'n_leaves', 'splits', 'distinct_feature_indx', 'regions'].
            info['depth']: the depth of the tree; info['n_leaves']: the number of leaves; info['splits]: a list of all splits happening in the tree recorded in the form of (feature_index,threshold);
            info['distinct_feature_indx']: records indices of distinct features with respect to which splittings occur in the ascending order;
            info['regions'] is a list of dictionaries, each describing the region determined by a leaf. The keys for each element of info['regions'] are 'value', 'weight', 'probability', and the indices of features which should satisfy non-trivial bounds in the region.
        '''
        # Initializing the dictionary
        info = {}
    
        # The first two keys are easy:
        info['depth'] = len(tree_structure['splits'])
        info['n_leaves'] = 2**info['depth']
    
        # Initializing the next two keys:
        info['splits'] = []
        info['distinct_feature_indx'] = []
        for split in tree_structure['splits']:                  # Each element of tree_structure['splits'] describes a splitting that takes place across an entire level.
            if split['float_feature_index'] not in info['distinct_feature_indx']:
                info['distinct_feature_indx'] += [split['float_feature_index']]
            info['splits'] += [(split['float_feature_index'],split['border'])]
    
        # It remains to compute info['region'], a list comprised of one dictionary per region. 
        # Initializing:
        info['regions'] = []
        for i in range(2**info['depth']):
            # Constructing the dictionary describing this region:
            region = {}
            region['value'] = tree_structure['leaf_values'][i]
            region['weight'] = tree_structure['leaf_weights'][i]
            
            # Initializing the keys that describe bounds for each feature.
            for feature_index in info['distinct_feature_indx']:
                region[feature_index] = [-float('inf'),float('inf')]
                
            expansion='{0:b}'.format(i)                         # The binary expansion of i which is the index of the leaf/region under consideration.
            while len(expansion)<info['depth']:                 # (An integer from [0,2**depth-1], we want len(expansion)=depth.) 
                expansion = '0'+expansion
                                                                
            for j in range(info['depth']):                      # The leftmost characters of the expansion are determined by top splits near the root which are encoded by the rightmost entries of info['splits'].
                feature_index = info['splits'][-j-1][0]              
                threshold = info['splits'][-j-1][1]             # (Keep in mind that splits closer to the root appear at the end of tree_structure['splits']).
                
                if expansion[j] == '0':                         # Meaning we go to the left since feature_value<threshold.
                    region[feature_index] = Parser._modify_interval(region[feature_index],threshold,'upper')
                else:                                              # Meaning we go to the left since feature_value>threshold.
                    region[feature_index] = Parser._modify_interval(region[feature_index],threshold,'lower')
            
            # Adding the dictionary constructed for this region to info['regions'].
            info['regions'] += [region]
            
        # Adding a key for probability to each dictionary from info['regions']
        total_weight=0
        for region in info['regions']:
            total_weight += region['weight']
        for region in info['regions']:
            region['probability'] = region['weight']/total_weight
        
        return info
    #########################################################################################################
    
    
    ################################## Parsing a LightGBM Tree ##############################################
    @staticmethod
    def _retrieve_lgbm(tree_structure):        
        '''
        Parses a tree of a trained LightGBM ensemble.
        
        Input
        -----
        The input tree_structure is a dictionary describing a tree of a trained LightGBM ensemble. 
        (It has a nested structure: the feature and the threshold w.r.t. which the tree splits at the root can be recovered from keys 'split_feature' and 'threshold' while the left and right subtrees can be recovered from keys 'left_child' and 'right_child'.)
        
        Output
        ------
        A dictionary info with keys info.keys()=['depth', 'n_leaves', 'splits', 'distinct_feature_indx', 'regions'].
            info['depth']: the depth of the tree; info['n_leaves']: the number of leaves; info['splits]: a list of all splits happening in the tree recorded in the form of (feature_index,threshold);
            info['distinct_feature_indx']: records indices of distinct features with respect to which splittings occur in the ascending order;
            info['regions'] is a list of dictionaries, each describing the region determined by a leaf. The keys for each element of info['regions'] are 'value', 'weight', 'probability', and the indices of features which should satisfy non-trivial bounds in the region.
        '''                                                            
        # Initializing the dictionary
        info = {}
        info['depth'] = 0 
        info['n_leaves'] = 1
        info['splits'] = []
        info['distinct_feature_indx'] = []
        info['regions'] = []
    
        # A single node is considered to be a tree of depth 0 with just 1 leaf. 
        if 'split_feature' not in tree_structure.keys():
            info['value'] = tree_structure['leaf_value']
            info['weight'] = tree_structure['leaf_weight']
            return info 
    
        # The first splitting happening at the root:
        feature_indx_root = tree_structure['split_feature']
        threshold_root = tree_structure['threshold']
    
        # The left sub-tree branching from the root:
        subtree_left = tree_structure['left_child']
        info_left = Parser._retrieve_lgbm(subtree_left)          # The function is recursive.    
    
        # The right sub-tree branching from the root: 
        subtree_right = tree_structure['right_child']           # The function is recursive.
        info_right = Parser._retrieve_lgbm(subtree_right)
    
        # Computing the depth:
        info['depth'] = 1+max(info_left['depth'],info_right['depth'])

        # Computing the number of leaves:
        info['n_leaves'] = info_left['n_leaves']+info_right['n_leaves']

        # Recording the splittings:
        info['splits'] = [(feature_indx_root,threshold_root)]+info_left['splits']+info_right['splits']

        # Recording the indices of the different features appearing in the tree in ascending order:
        for split in info['splits']:
            if split[0] not in info['distinct_feature_indx']:
                info['distinct_feature_indx'] += [split[0]]
        info['distinct_feature_indx'].sort()

        # info['regions'] is built recursively via calling the function for left and right subtrees. 
        # One should condition on if the left or right subtree is a single node.
        # Keep in mind that at a node if feature<=threshold, we go to left, otherwise to right. 
        # (The value of the key 'decision_type' of tree_structure should be '<=', the same true for dictionaries embedded in it.) 
        # The auxiliary function _modify_interval is used to update a range obtained from a subtree according to the threshold at the root. 
        if info_left['depth'] == 0 and info_right['depth'] == 0:
            info['regions'] = [{'value':tree_structure['left_child']['leaf_value'],'weight':tree_structure['left_child']['leaf_weight'],
                                feature_indx_root:[-float('inf'),threshold_root]}
                                ,{'value':tree_structure['right_child']['leaf_value'],'weight':tree_structure['right_child']['leaf_weight'],
                                feature_indx_root:[threshold_root,float('inf')]}]
        elif info_left['depth'] == 0 and info_right['depth'] != 0:
            info['regions'] = [{'value':tree_structure['left_child']['leaf_value'],'weight':tree_structure['left_child']['leaf_weight'],
                                feature_indx_root:[-float('inf'),threshold_root]}]
            for region in info_right['regions']:
                if feature_indx_root not in region.keys():
                    region[feature_indx_root] = [threshold_root,float('inf')]
                elif Parser._modify_interval(region[feature_indx_root],threshold_root,'lower') != None:
                    region[feature_indx_root] = Parser._modify_interval(region[feature_indx_root],threshold_root,'lower')
            info['regions'] += info_right['regions']        
        elif info_left['depth'] != 0 and info_right['depth'] == 0:
            for region in info_left['regions']:
                if feature_indx_root not in region.keys():
                    region[feature_indx_root] = [-float('inf'),threshold_root]
                elif Parser._modify_interval(region[feature_indx_root],threshold_root,'upper') != None:
                    region[feature_indx_root] = Parser._modify_interval(region[feature_indx_root],threshold_root,'upper')
            info['regions'] = info_left['regions']   
            info['regions'] += [{'value':tree_structure['right_child']['leaf_value'],'weight':tree_structure['right_child']['leaf_weight'],
                                feature_indx_root:[threshold_root,float('inf')]}]
        else:
            for region in info_left['regions']:
                if feature_indx_root not in region.keys():
                    region[feature_indx_root]=[-float('inf'),threshold_root]
                elif Parser._modify_interval(region[feature_indx_root],threshold_root,'upper') != None:                    # None means the region is vacuous or degenerate. 
                    region[feature_indx_root] = Parser._modify_interval(region[feature_indx_root],threshold_root,'upper')
            info['regions'] = info_left['regions']
            for region in info_right['regions']:
                if feature_indx_root not in region.keys():
                    region[feature_indx_root] = [threshold_root,float('inf')]
                elif Parser._modify_interval(region[feature_indx_root],threshold_root,'lower') != None:                    # None means the region is vacuous or degenerate.
                    region[feature_indx_root] = Parser._modify_interval(region[feature_indx_root],threshold_root,'lower')
            info['regions'] += info_right['regions']        
    
    
        # Adding a key for probability to each dictionary from info['regions']
        total_weight = 0
        for region in info['regions']:
            total_weight += region['weight']
        for region in info['regions']:
            region['probability'] = region['weight']/total_weight
        return info
    #########################################################################################################
    
    
    ################################## Parsing an XGBoost Tree ##############################################
    @staticmethod
    def _retrieve_xgb(tree_df):
        '''
        Parses a tree of a trained XGBoost ensemble.
        
        Input
        -----
        The input tree_structure is a dataframe describing a tree of a trained XGBoost ensemble.
        (Each row of which describes a node (either internal or a leaf) of the tree. 
        The splits can be recovered columns 'Feature' and 'Split' while for the values and the number of training instances at leaves we utilize columns 'Gain' and 'Cover'. To column 'ID' is used to recovering a node's parent.)
        
        Output
        ------
        A dictionary info with keys info.keys()=['depth', 'n_leaves', 'splits', 'distinct_feature_indx', 'regions'].
            info['depth']: the depth of the tree;  info['n_leaves']: the number of leaves; info['splits]: a list of all splits happening in the tree recorded in the form of (feature_index,threshold);
            info['distinct_feature_indx']: records indices of distinct features with respect to which splittings occur in the ascending order;
            info['regions'] is a list of dictionaries, each describing the region determined by a leaf. The keys for each element of info['regions'] are 'value', 'weight', 'probability', and the indices of features which should satisfy non-trivial bounds in the region.
        '''                                                           
        split_rows = tree_df.index[tree_df['Feature']!='Leaf'].to_list()         # rows corresponding to splits
        leaf_rows = tree_df.index[tree_df['Feature']=='Leaf'].to_list()          # rows corresponding to leaves

        # Initializing the dictionary
        info = {}
        info['n_leaves'] = len(leaf_rows)
        info['splits'] = []
        info['distinct_feature_indx'] = []
    
        for i in split_rows:                                                     # Recording splits and distinct features                                        
            feature_index = int(tree_df['Feature'][i][1:])
            threshold = tree_df['Split'][i]   
            info['splits'] += [(feature_index,threshold)]
            if feature_index not in info['distinct_feature_indx']:
                info['distinct_feature_indx'] += [feature_index]
            
            
        info['regions'] = []                                                     # The region corresponding to a leaf is recovered through calling an auxiliary function.
        depth = 0     
        total_weight = 0                                                         # This will be used to obtain probabilities of regions. 
        for i in leaf_rows:  
            region = Parser._retrieve_xgb_region(tree_df,i)
            depth = max(depth,region['depth'])                                   # We compare depths of all regions to obtain tree's depth.
            total_weight += region['weight']
            del region['depth']                                                  # Deleting the auxiliary key for depth
            info['regions'] += [region]
            
        for region in info['regions']:
            region['probability'] = region['weight']/total_weight

        info['depth'] = depth
        return info
    
    # Auxiliary function: Returns the region corresponding to the leaf appearing on a given row of the data frame. 
    def _retrieve_xgb_region(tree_df,leaf_row):                        
        region={}
        region['value'] = tree_df['Gain'][leaf_row]
        region['weight'] = tree_df['Cover'][leaf_row]
        
        row_index = leaf_row
        depth = 0
        while True:                                                              # Going up from the leaf until we reach the root. 
            if row_index == 0:                                                   # In this case, we have reached the root, there is nothing more to do. 
                break
            depth += 1
            if len(tree_df.index[tree_df['Yes'] == tree_df['ID'][row_index]].to_list())>0: 
                kind='upper'                                                     # feature value<threshold on the Yes column
                row_index = tree_df.index[tree_df['Yes'] == tree_df['ID'][row_index]].to_list()[0]
            else:                                                                # feature value>threshold on the No column
                kind='lower'
                row_index = tree_df.index[tree_df['No'] == tree_df['ID'][row_index]].to_list()[0]
            feature_index = int(tree_df['Feature'][row_index][1:])               # feature_index and threshold for the parent (to be considered in the next iteration)
            threshold = tree_df['Split'][row_index]
            if feature_index in region.keys():                                   # If the feature already appears, we modify the interval; otherwise, we add it as a new key. 
                region[feature_index] = Parser._modify_interval(region[feature_index],threshold,kind)
            else:
                if kind == 'upper':
                    region[feature_index] = [-float('inf'),threshold]
                else:
                    region[feature_index] = [threshold,float('inf')]
        
        region['depth'] = depth                              
        return region
    #########################################################################################################

    
    ################################## An Auxiliary Function ################################################
    @staticmethod
    def _modify_interval(interval,bound,kind):
        '''
        Returns the intersection of interval with <=bound when kind=='upper' and with >=bound when kind=='lower'.
        Returns None if the intersection is empty or degenerate.
        '''
        if interval==None:                                      # Nothing to modify if the interval is empty to begin with. 
            return None
        if kind=='upper':
            if interval[0]>=bound:
                return None
            else:
                interval[1]=min(interval[1],bound)
        else:
            if interval[1]<=bound:
                return None
            else:
                interval[0]=max(interval[0],bound)
        return interval            
    #########################################################################################################