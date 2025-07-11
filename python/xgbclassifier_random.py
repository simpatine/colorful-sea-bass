import argparse
import json
import math
import os
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s\t%(levelname)s\t:%(message)s",encoding='utf-8', level=logging.INFO)

import copy
import ast
from functools import partial
import numpy as np
import pandas as pd
import xgboost as xgb
from numpy import ndarray
import sklearn.metrics as mt
from sklearn.model_selection import train_test_split
from xgboost import Booster
from concurrent.futures import ProcessPoolExecutor

xgb.config_context(nthread=32)

def read_feature_list(selection_file):
    features = pd.read_csv(selection_file, header=None)
    logging.info(f"Read {len(features)} features to select")
    return features.iloc[:, 0].tolist()


def print_dataset_stats(X, y, label):
    logging.info(f"\tData points: {X.shape[0]}")
    logging.info(f"\t\tnumber of features: {X.shape[1]}")
    if len(y[label].unique()) == 2:
        logging.info(f"\t\tlabel(0) counts: {(y[label] == 0).sum() / len(y[label]) * 100 : .2f} %")
        logging.info(f"\t\tlabel(1) counts: {(y[label] == 1).sum() / len(y[label]) * 100 : .2f} %")


def group_by_chromosome(weigths, gains):
    keys = weigths.keys()
    counts_group = [0] * 24
    weigths_group = [0] * 24
    gains_group = [0] * 24
    for key in keys:
        chromosome = int(key[13:15]) - 1  # e.g. CAJNNU010000001.1:4119343
        counts_group[chromosome] += 1
        weigths_group[chromosome] += weigths[key]
        gains_group[chromosome] += gains[key]

    return counts_group, weigths_group, gains_group


def group_by_region(weights, gains, regions_folder):
    # List of CSV file paths
    regions_files = ["Br_mock_broad.csv", "Br_NNV_broad.csv", "Hk_mock_broad.csv", "Hk_NNV_broad.csv", "broad.csv",
                     "Br_mock_narr.csv", "Br_NNV_narr.csv", "Hk_mock_narr.csv", "Hk_NNV_narr.csv", "narrow.csv"]

    # Create a list to store lists of strings from each CSV file
    total_gain = gains.loc[:, 0].sum()

    total_counts = weights.loc[:, 0].sum()
    features = set(weights.index)

    counts_dic = {}
    weights_dic = {}
    gains_dic = {}

    logging.info("Peaks\tCounts\tWeights\tGains")
    for file in regions_files:
        current_list = pd.read_csv(regions_folder + file).iloc[:, 0].tolist()
        common_strings = features.intersection(current_list)

        grouped_gains = gains.loc[list(common_strings), 0]
        grouped_counts = weights.loc[list(common_strings), 0]

        if len(features) == 0: continue

        counts_dic[file] = len(common_strings) / len(features)
        weights_dic[file] = grouped_counts.sum() / total_counts
        gains_dic[file] = grouped_gains.sum() / total_gain
        logging.info(f"{file}\t"
              f"{gains_dic[file] * 100: .2f} %\t"
              f"{counts_dic[file] * 100: .2f} %\t"
              f"{weights_dic[file] * 100: .2f}%")
        if file == "broad.csv":
            logging.info("\n")

    return regions_files, counts_dic, weights_dic, gains_dic


def data_ensemble(gains, data_ensemble_file):
    features = set(gains.index.values)

    ensemble = pd.read_csv(data_ensemble_file, index_col=0, header=0)
    intersection_features = list(features.intersection(ensemble.index.values))
    logging.info(f"#features in the ensemble: {len(intersection_features)}")

    ensemble = ensemble.loc[intersection_features, :]
    ensemble["gain"] = gains.loc[intersection_features, "gain"]
    ensemble.sort_values(by="gain", inplace=True, ascending=False)
    ensemble.to_csv("ensemble.csv")

    if len(intersection_features) > 0: # TODO fix bug here
        info_funct = ensemble["funct"].value_counts()
        info_funct["gain"] = ensemble.groupby("funct")["gain"].sum() # KeyError: 'Column not found: gain'

        info_n_tissue = ensemble["n_tissue"].value_counts()
        info_n_tissue["gain"] = ensemble.groupby("funct")["gain"].sum()

        logging.info(info_funct)
        logging.info("\n")
        logging.info(info_n_tissue)
        return intersection_features, info_funct, info_n_tissue
    else:
        logging.warning("No features in the data ensemble!!!")
        return intersection_features, pd.DataFrame(), pd.DataFrame()

def parse_line(line: str):
    parts = line.strip().split(',')
    idx    = parts[0]
    arr    = np.fromiter(map(int, parts[1:]), dtype=np.int8)
    return idx, arr

class XGBoostVariant:
    bst: Booster
    num_trees: int
    best_it: int
    best_score: float

    dtrain: xgb.DMatrix | xgb.QuantileDMatrix
    dvalidation: xgb.DMatrix | xgb.QuantileDMatrix
    dtest: xgb.DMatrix | xgb.QuantileDMatrix

    y_pred: ndarray
    y_test: ndarray

    target: str
    features = None
    importance_weights = None
    importance_gains = None

    def __init__(self, model_name, num_trees, max_depth, min_child_weight, eta, early_stopping,
                 method, objective, base_score, grow_policy, validation, train_set_file, sample_bytree,
                 sample_by_level, sample_bynode, num_parallel_trees, data_ensemble_file, features_sets_dir,
                 random_state
                 ):
        self.base_score = base_score
        self.estimated_base_score = None
        self.y_train_mean = .5
        self.matthews = None
        self.pcoeff_correct = None
        self.pcoeff = None
        self.mae = None
        self.rmse = None
        self.features_sets_dir = features_sets_dir
        self.data_ensemble_file = data_ensemble_file
        self.num_parallel_trees = num_parallel_trees
        self.auc = None
        self.f1 = None
        self.accuracy = None
        self.num_features = None
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.eta = eta
        self.early_stopping = early_stopping
        self.by_tree = sample_bytree
        self.by_node = sample_bynode
        self.by_level = sample_by_level
        self.random_state = random_state
        self.model_name = model_name
        self.num_trees = num_trees
        self.train_frac = .8
        self.method = method
        self.objective = objective  # binary:logistic or reg:logistic
        self.grow_policy = grow_policy

        if early_stopping is None:
            self.early_stopping = self.num_trees

        self.validation = validation
        self.train_set_file = train_set_file

        logging.info(f"Using XGBoost version {xgb.__version__}")

    def read(self, data_file):
        logging.info("Loading features...")
        start_t = time.time()
        with open(data_file, 'r') as f:
            column_names = next(f).strip().split(',')[1:]

            with ProcessPoolExecutor() as pool:
                results = pool.map(parse_line, f, chunksize=10)

                idxs, rows = zip(*results)
                indexes   = list(idxs)
                data_rows = list(rows)

            data_array = np.array(data_rows)
            data = pd.DataFrame(data_array, columns=column_names, index=indexes)
        stop_t = time.time()
        logging.info(f"Done in {stop_t - start_t : .2f} s.")

        return data

    def subsample(self, data, subsampling, subsampling_ratios = None, iterations = 1):

        sampled_data = []

        logging.info("Subsampling features...")

        if subsampling_ratios is None:
            start_shuffle_t = time.time()
            for i in range(iterations):
                data_n = subsampling(data)
                sampled_data.append([copy.deepcopy(self), data_n, None, i])

            stop_shuffle_t = time.time()
            logging.info(f"Done in {stop_shuffle_t - start_shuffle_t : .2f} s")
            return sampled_data

        if type(subsampling_ratios) != list:
            subsampling_ratios = [subsampling_ratios]

        start_shuffle_t = time.time()
        for subsample_ratio in subsampling_ratios:
            for i in range(iterations):
                data_n = subsampling(data,subsample_ratio)
                sampled_data.append([copy.deepcopy(self), data_n, subsample_ratio, i])

        stop_shuffle_t = time.time()
        logging.info(f"Done in {stop_shuffle_t - start_shuffle_t : .2f} s")

        return sampled_data


    def transformation(self, data, target_file):

        validation = self.validation
        train_set_file = self.train_set_file

        self.features = list(data.columns)

        labels = pd.read_csv(target_file, header=0, index_col=0)
        self.target = labels.columns[0]
        if train_set_file is None:
            if validation:
                X_train, X_test, y_train, y_test = train_test_split(data,
                                                                    labels,
                                                                    train_size=self.train_frac,
                                                                    random_state=42
                                                                    )
                X_test, X_validation, y_test, y_validation = train_test_split(X_test,
                                                                              y_test,
                                                                              train_size=.5,
                                                                              random_state=42
                                                                              )
            else:
                X_train, X_test, y_train, y_test = train_test_split(data,
                                                                    labels,
                                                                    train_size=self.train_frac,
                                                                    random_state=42
                                                                    )
        else:
            train_cluster = pd.read_csv(self.train_set_file, header=0)["id"].values.tolist()

            X_train = data.loc[train_cluster]
            y_train = labels.loc[train_cluster]  # Series
            y_train = pd.DataFrame(y_train)

            X_test = data.drop(train_cluster)
            y_test = labels.drop(train_cluster)  # Series
            y_test = pd.DataFrame(y_test)

            self.train_frac = len(X_train) / (len(X_train) + len(X_test))

        self.y_train_mean = y_train.mean().iloc[0]

        # self.dtrain = xgb.DMatrix(X_train, y_train)
        if self.method == "hist":
            self.dtrain = xgb.QuantileDMatrix(X_train, y_train)
        else:
            self.dtrain = xgb.DMatrix(X_train, y_train)

        if validation:
            if self.method == "hist":
                self.dvalidation = xgb.QuantileDMatrix(X_validation, y_validation, ref = self.dtrain)
            else:
                self.dvalidation = xgb.DMatrix(X_validation, y_validation)
        else:
            self.dvalidation = None

        self.y_test = y_test
        if self.method == "hist":
            self.dtest = xgb.QuantileDMatrix(X_test, ref = self.dtrain)
        else:
            self.dtest = xgb.DMatrix(X_test)


    def read_datasets(self, data_file, target_file, subsampling = None):
        """
Parameters
-----------
data_file: str/Path
    Path of the genome data
target_file: str/Path
    Path of the target data
subsampling: function
    function for subsampling data
        """
        validation = self.validation
        train_set_file = self.train_set_file

        start_t = time.time()
        logging.info("Loading features...")
        start_t = time.time()
        with open(data_file, 'r') as f:
            column_names = next(f).strip().split(',')[1:]

            with ProcessPoolExecutor() as pool:
                results = pool.map(parse_line, f, chunksize=10)

                idxs, rows = zip(*results)
                indexes   = list(idxs)
                data_rows = list(rows)

            data_array = np.array(data_rows)
            data = pd.DataFrame(data_array, columns=column_names, index=indexes)
        stop_t = time.time()
        logging.info(f"Done in {stop_t - start_t : .2f} s.")

        if subsampling is not None:
            logging.info("Subsampling features...")
            start_shuffle_t = time.time()

            data = subsampling(data)

            stop_shuffle_t = time.time()
            logging.info(f"Done in {stop_shuffle_t - start_shuffle_t : .2f} s")

        self.features = list(data.columns)

        logging.info("Reading targets...")
        labels = pd.read_csv(target_file, header=0, index_col=0)
        self.target = labels.columns[0]
        logging.info(f"Target is {self.target}")
        logging.info("Done.")

        logging.info("Splitting the datasets...")
        if train_set_file is None:
            if validation:
                X_train, X_test, y_train, y_test = train_test_split(data,
                                                                    labels,
                                                                    train_size=self.train_frac,
                                                                    random_state=42
                                                                    )
                X_test, X_validation, y_test, y_validation = train_test_split(X_test,
                                                                              y_test,
                                                                              train_size=.5,
                                                                              random_state=42
                                                                              )
            else:
                X_train, X_test, y_train, y_test = train_test_split(data,
                                                                    labels,
                                                                    train_size=self.train_frac,
                                                                    random_state=42
                                                                    )
        else:
            logging.info(f"Reading training set IDs...")
            train_cluster = pd.read_csv(self.train_set_file, header=0)["id"].values.tolist()

            X_train = data.loc[train_cluster]
            y_train = labels.loc[train_cluster]  # Series
            y_train = pd.DataFrame(y_train)

            X_test = data.drop(train_cluster)
            y_test = labels.drop(train_cluster)  # Series
            y_test = pd.DataFrame(y_test)

            self.train_frac = len(X_train) / (len(X_train) + len(X_test))

        logging.info("Done.\n")

        self.y_train_mean = y_train.mean().iloc[0]

        start_transf_t = time.time()
        logging.info("Stats (train data):")
        print_dataset_stats(X_train, y_train, self.target)
        logging.info(f"\tmean(y_train) = {self.y_train_mean}")
        logging.info("Transforming X_train and y_train into DMatrices...")
        # self.dtrain = xgb.DMatrix(X_train, y_train)
        if self.method == "hist":
            self.dtrain = xgb.QuantileDMatrix(X_train, y_train)
        else:
            self.dtrain = xgb.DMatrix(X_train, y_train)
        logging.info("Done.\n")

        if validation:
            logging.info("Stats (validation data):")
            print_dataset_stats(X_validation, y_validation, self.target)
            logging.info("Transforming X_validation and y_validation into DMatrices...")
            if self.method == "hist":
                self.dvalidation = xgb.QuantileDMatrix(X_validation, y_validation, ref = self.dtrain)
            else:
                self.dvalidation = xgb.DMatrix(X_validation, y_validation)
        else:
            self.dvalidation = None

        logging.info("Stats (test data):")
        print_dataset_stats(X_test, y_test, self.target)
        logging.info("Transforming X_test into DMatrices...")
        self.y_test = y_test
        if self.method == "hist":
            self.dtest = xgb.QuantileDMatrix(X_test, ref = self.dtrain)
        else:
            self.dtest = xgb.DMatrix(X_test)

        stop_transf_t = time.time()
        logging.info(f"Transformation time: {stop_transf_t - start_transf_t : .2f} s")
        end_t = time.time()
        logging.info(f"Read time {end_t - start_t : .2f} s")

    def set_weights(self, weights=None, equal_weight=False):
        # feature weights TODO fix random order with dictionary on weights!!!
        """
        if feature_weights is not None:
            print("Setting weights...")
            assert len(feature_weights) == self.dtrain.num_col()
            self.dtrain.set_info(feature_weights=feature_weights)
        """
        if weights is None:
            weights = self.importance_weights

        fw = []
        for feature in self.features:
            w = weights.get(feature, 0)
            if equal_weight and w > 0:
                w = 1
            fw.append(w)

        self.dtrain.set_info(feature_weights=fw)

    def fit(self, params=None, evals=None, cuda=False):
        if self.dtrain is None:
            raise Exception("Need to load training datasets first!")

        if params is None:
            params = {"verbosity": 1, "device": "cpu", "tree_method": self.method,
                      "objective": self.objective, "grow_policy": self.grow_policy,
                      "seed": 42, "nthread" : -1,
                      "eta": self.eta, "max_depth": self.max_depth, "min_child_weight": self.min_child_weight
                      }

            if self.by_tree < 1:
                params["colsample_bytree"] = self.by_tree
            if self.by_node < 1:
                params["colsample_bynode"] = self.by_node
            if self.by_level < 1:
                params["colsample_bylevel"] = self.by_level

            if self.num_parallel_trees > 1:
                params["num_parallel_tree"] = self.num_parallel_trees
                if not (self.by_tree < 1 or self.by_node < 1 or self.by_level < 1):
                    logging.warning(f"You need to add randomness to your Random Forest!")

            if self.base_score is not None:
                params["base_score"] = self.base_score
        if evals is None:
            if self.dvalidation is None:
                evals = [(self.dtrain, "training")]
            else:
                evals = [(self.dtrain, "training"), (self.dvalidation, "validation")]

        # use CUDA if available
        if cuda:
            params["device"] = "cuda"
        else:
            params["nthread"] = 0
        try:
            logging.info(f"Trying with {params['device']}")
            self.bst = xgb.train(params=params,
                                 dtrain=self.dtrain,
                                 num_boost_round=self.num_trees,
                                 evals=evals,
                                 verbose_eval=0,
                                 early_stopping_rounds=self.early_stopping
                                 )
            logging.info(f"Done training accelerated with {params['device']}")
        except Exception as e:
            logging.error(e)
            logging.error("Error during training. Exiting...")
            exit(-1)

        # update number of trees in case of early stopping
        self.num_trees = self.bst.num_boosted_rounds()

        # best values
        self.best_it = self.bst.best_iteration
        self.best_score = self.bst.best_score

        # features importance
        self.importance_weights = self.bst.get_score(importance_type="weight")
        self.importance_gains = self.bst.get_score(importance_type="gain")

        # save model
        self.bst.save_model(f"{self.model_name}.json")

    def predict(self, iteration_range=None):
        if iteration_range is None:
            iteration_range = (0, self.best_it)

        self.y_pred = self.bst.predict(self.dtest, iteration_range=iteration_range)

    def print_stats(self, stdout=False):
        logging.info("\n+++ Prediction stats +++")

        logging.info(f"Best score: {self.best_score}")
        logging.info(f"Best iteration: {self.best_it}")

        if "bin" in self.objective:  # classification
            conf_mat = mt.confusion_matrix(self.y_test, self.y_pred, labels=[0,1])
            true_neg = conf_mat[0][0]
            true_pos = conf_mat[1][1]
            false_neg = conf_mat[1][0]
            false_pos = conf_mat[0][1]

            assert (true_pos + false_neg) == sum(self.y_test[self.target])
            assert (true_neg + false_pos) == len(self.y_test[self.target]) - sum(self.y_test[self.target])
            assert (true_neg + true_pos + false_neg + false_pos) == len(self.y_test[self.target])

            logging.info(f"TN={true_neg}\tFP={false_pos}")
            logging.info(f"FN={false_neg}\tTP={true_pos}")

            # accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

            self.accuracy = mt.accuracy_score(self.y_test, self.y_pred)
            self.f1 = mt.f1_score(self.y_test, self.y_pred)
            try:
                self.auc = mt.roc_auc_score(self.y_test, self.y_pred)
            except ValueError:
                self.auc = None
            self.matthews = mt.matthews_corrcoef(self.y_test, self.y_pred)

            if stdout:
                print(f"{self.subsample_ratio : .10f};{self.accuracy * 100 : .3f};{self.f1 * 100 : .3f};{self.random_state}")

            logging.info(f"Accuracy = {self.accuracy * 100 : .3f} %")
            logging.info(f"f1 = {self.f1 * 100 : .3f} %")
            logging.info(f"Matthews = {self.matthews}")
            try:
                logging.info(f"ROC_AUC = {self.auc * 100 : .3f} %")
            except TypeError:
                logging.info("ROC_AUC = None")

        else:  # regression
            self.mae = mt.mean_absolute_error(self.y_test, self.y_pred)
            self.rmse = np.sqrt(mt.mean_squared_error(self.y_test, self.y_pred))
            self.pcoeff = np.corrcoef(self.y_test, self.y_pred, rowvar=False)[0][1]
            reliability = 0.999
            self.pcoeff_correct = np.sqrt((self.pcoeff ** 2) * reliability)

            logging.info(f"MAE = {self.mae : .2f}")
            logging.info(f"RMSE = {self.rmse : .2f}")
            logging.info(f"pearson = {self.pcoeff}")
            logging.info(f"pearson_correct = {self.pcoeff_correct}")

        # save y_pred
        predictions = self.y_test
        predictions["pred"] = self.y_pred
        predictions.to_csv("predictions.csv")

        self.estimated_base_score = json.loads(self.bst.save_config())["learner"]["learner_model_param"][
            "base_score"
        ]
        # print top features
        print_num_feat = 10
        importance = sorted(self.importance_gains.items(), key=lambda item: item[1], reverse=True)
        self.num_features = len(importance)
        logging.info(f"Top {print_num_feat}/{self.num_features} features (gains):")
        logging.info(importance[:print_num_feat])

    def write_stats(self, stats_file="stats.csv"):
        logging.info(f"Writing stats to {stats_file}")
        with open(stats_file, 'w') as stats:
            stats.write(f"method name,{self.model_name}\n")
            stats.write(f"algorithm,{self.method}\n")
            # stats.write(f"training set,{self.train_frac}\n") # TODO: check why this is sus
            # stats.write(f"training set file,{self.train_set_file}\n") # TODO: check why this is sus
            stats.write(f"validation set,{self.validation}\n")
            stats.write(f"uniform over chromosomes, {args.uniform_over_chromosomes}")
            # stats.write(f"feature set,{self.features_set_file}\n")
            stats.write(f"features available,{len(self.features)}\n")
            stats.write(f"early stopping,{self.early_stopping}\n")
            stats.write(f"trees,{self.num_trees}\n")
            stats.write(f"eta,{self.eta}\n")
            stats.write(f"max depth,{self.max_depth}\n")
            stats.write(f"grow_policy,{self.grow_policy}\n")
            stats.write(f"parallel trees,{self.num_parallel_trees}\n")
            stats.write(f"sampling by tree,{self.by_tree}\n")
            stats.write(f"sampling by level,{self.by_level}\n")
            stats.write(f"sampling by node,{self.by_node}\n")

            stats.write("\n")

            stats.write(f"estimated base score,{self.estimated_base_score}\n")
            stats.write(f"selected features,{self.num_features}\n")
            stats.write(f"accuracy,{self.accuracy}\n")
            stats.write(f"f1,{self.f1}\n")
            stats.write(f"ROC AUC,{self.auc}\n")
            stats.write(f"Matthews,{self.matthews}\n")
            stats.write(f"Pearson,{self.pcoeff}\n")
            stats.write(f"Pearson_correct,{self.pcoeff_correct}\n")
            stats.write(f"MAE,{self.mae}\n")
            stats.write(f"RMSE,{self.rmse}\n")
            stats.write(f"best iteration,{self.best_it}\n")
            stats.write(f"best score,{self.best_score}\n")
            stats.write(f"tree created,{self.bst.num_boosted_rounds()}\n")

            stats.write("\n")

            stats.write("CHR,count,weight,gain\n")
            counts, weights, gains = group_by_chromosome(self.importance_weights, self.importance_gains)
            for i in range(24):
                stats.write(f"{i + 1},{counts[i]},{weights[i]},{gains[i]}\n")

            stats.write("\n")

            stats.write("set,count,weight,gain\n")
            peaks, counts, weights, gains = group_by_region(
                pd.DataFrame(self.importance_weights, index=pd.Index([0])).T,
                pd.DataFrame(self.importance_gains, index=pd.Index([0])).T,
                self.features_sets_dir
                )
            for peak in peaks:
                try: stats.write(f"{peak},{counts[peak]},{weights[peak]},{gains[peak]}\n")
                except: continue
            stats.write("\n")

            intersection_features, info_funct, info_n_tissue = data_ensemble(gains=pd.DataFrame(self.importance_gains, index=pd.Index(["gain"])).T, data_ensemble_file=self.data_ensemble_file)
            stats.write(f"features in the ensemble,{len(intersection_features)}\n")

            stats.write("\n")

            stats.write("funct,count\n")
            for i in info_funct.index:
                stats.write(f"{i},{info_funct[i]}\n")
            stats.write("\n")

            stats.write("n_tissue,count\n")
            for i in info_n_tissue.index:
                stats.write(f"{i},{info_n_tissue[i]}\n")
            stats.write("\n")

            k = 10
            if len(self.importance_gains) < k:
                k = len(self.importance_gains)
            stats.write(f"Top {k} gains\n")
            top_gains = sorted(self.importance_gains.items(), key=lambda x: x[1], reverse=True)[:k]
            for g in top_gains:
                stats.write(f"{g[0]},{g[1]}\n")

    def plot_trees(self, tree_set=None, tree_name=None, render=False):
        logging.info("Printing trees...")
        if tree_set is None:
            tree_set = range(self.num_trees)

        if tree_name is None:
            tree_name = self.model_name

        logging.info("Done.")

    def write_importance(self, filename):
        with open(filename + ".weights.csv", 'w') as importance_file:
            for item in sorted(self.importance_weights.items(), key=lambda x: x[1]):
                importance_file.write(f"{item[0]}, {item[1]}\n")

        with open(filename + ".gains.csv", 'w') as importance_file:
            for item in sorted(self.importance_gains.items(), key=lambda x: x[1]):
                importance_file.write(f"{item[0]}, {item[1]}\n")


def subsample_standard(data, subsample_ratio = 1):
    n_columns = len(data.columns)
    select = np.zeros(n_columns, dtype=bool)
    n_sampled = int(subsample_ratio * n_columns)

    select[:n_sampled] = 1
    np.random.shuffle(select)

    return data.loc[:, select]

def subsample_uniform_chromosomes(data, subsample_ratio = 1, chromosome_info = None):
    n_columns = len(data.columns)
    select = np.zeros(n_columns, dtype=bool)
    n_sampled = int(subsample_ratio * n_columns)

    if chromosome_info is None:
        chromosomes_list = np.array([name.split(":")[0] for name in data.columns])
        count = np.unique(chromosomes_list, return_counts=True)
        chromosomes_count = {count[0][i] : count[1][i] for i in range(len(count[0]))}
    else:
        chromosomes_list, chromosomes_count = chromosome_info

    n_sampled_chromo = n_sampled // len(chromosomes_count)

    for chromosome, count in chromosomes_count.items():
        chromosome_indices = np.where(chromosomes_list == chromosome)
        chromosome_select = np.zeros(count, dtype=bool)
        chromosome_select[:min(count, n_sampled_chromo)] = 1
        np.random.shuffle(chromosome_select)
        select[chromosome_indices] = chromosome_select

    return data.loc[:, select]

def subsample_annotated(data, snp_ids):
    selected_columns = list(snp_ids & set(data.columns))

    return data[selected_columns]

def parse_line_annotations(line: str, note_type=None, n_tissue=None):
    line_split = line.split(",")
    if (note_type is None or line_split[1] == note_type) and \
       (n_tissue is None or int(n_tissue) <= int(line_split[2]) < 10+int(n_tissue)):
        return line_split[0]
    else: "SNP not selected"

def read_annotations(file_path, note_type = None, n_tissue=None):
    """ """

    logging.info("Loading annotations...")
    start_t = time.time()
    with open(file_path, "r") as f:
        header = next(f)

        parse_line_annotations_wrapper = partial(parse_line_annotations, note_type=note_type, n_tissue=n_tissue)
        with ProcessPoolExecutor() as pool:
            snp_ids = pool.map(parse_line_annotations_wrapper, f, chunksize=10)
            snp_ids = set(snp_ids)
    snp_ids.discard("SNP not selected")
    logging.info(f"Done loading annotations in {time.time()-start_t : .2f}s")

    return snp_ids


def train(args, target_file, stdout = False):

    clf:XGBoostVariant = args[0]
    data = args[1]
    subsample_ratio = args[2]
    iteration = args[3]
    clf.random_state = iteration
    clf.subsample_ratio = subsample_ratio

    clf.transformation(data, target_file)
    clf.fit()
    clf.predict()
    clf.print_stats(stdout=stdout)
    #clf.write_importance(f"importance-{it}")
    clf.set_weights(equal_weight=True)

if __name__ == "__main__":
    logging.info("Starting now!")
    parser = argparse.ArgumentParser(description='XGBoost variant classifier')
    parser.add_argument("--model_name", type=str, default="default-model", help="Model name")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations")

    parser.add_argument("--dataset", type=str, default="features.csv",
                        help="Features csv file")
    parser.add_argument('--shuffle_features', default=True, action="store_true")
    parser.add_argument("--target", type=str, default="mortality.csv", help="Target csv file")
    parser.add_argument("--annotations", type=str, default=None, help="Annotations csv file")
    parser.add_argument("--annotation_type", type=str, default=None, help="Type of genes to be selected using the annotations")
    parser.add_argument("--n_tissue", type=int, default=None, help="n_tissue to be selected using the annotations")
    parser.add_argument('--validate', default=False, action="store_true")
    parser.add_argument("--select", type=str, default=None, help="List of feature to select")
    parser.add_argument("--subsample_ratios", type=str, default=None, help="Ratios of columns to select")
    parser.add_argument("--uniform_over_chromosomes", type=bool, default=False, help="Wether the subsampling is made uniform over all chromosomes")
    parser.add_argument("--cluster", type=str, default=None, help="Cluster for training")

    parser.add_argument("--method", type=str, default="exact", help="Tree method")
    parser.add_argument('--objective', default="binary:hinge", help="binary:hinge or binary:logistic or reg:squarederr...")
    parser.add_argument("--base_score", type=float, default=None, help="Starting value to affinate by the booster")
    parser.add_argument('--grow_policy', default="depthwise", help="depthwise or lossguide")
    parser.add_argument("--num_trees", type=int, default=100, help="Number of trees")
    parser.add_argument("--early_stopping", type=int, default=None, help="Stop after n non-increasing iterations")
    parser.add_argument("--max_depth", type=int, default=6, help="Max depth for trees")
    parser.add_argument("--eta", type=float, default=.2, help="Learning rate")
    parser.add_argument("--min_child_weight", type=int, default=1, help="Min child weigth, used to simplify the tree")

    parser.add_argument("--sample_bytree", type=float, default=1, help="Sample features by tree")
    parser.add_argument("--sample_bylevel", type=float, default=1, help="Sample features by level")
    parser.add_argument("--sample_bynode", type=float, default=1, help="Sample features by node")

    # random forest
    parser.add_argument("--num_parallel_trees", type=int, default=1, help="Number of parallel trees")

    # stats
    parser.add_argument("--data_ensemble", type=str,
                        default="/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/exclude-chr3/data_ensemble.csv",
                        help="Data ensemble cdv file with labeled SNPs (abs path)")
    parser.add_argument("--features_sets_dir", type=str, default="/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/exclude-chr3/features-sets/",
                        help="Directory with regions files (abs path)")

    parser.add_argument("--random-state", type=int, default=69, help="Number to use as random seed")
    parser.add_argument("--use-gpu", type=bool, default=False, help="Accelerate training with CUDA")
    parser.add_argument("--stdout-values", type=bool, default=False, help="Print minimal performance values to stdout")
    args = parser.parse_args()

    logging.info(args)

    if args.use_gpu and args.method != "hist":
        logging.error("When using CUDA device the only supported method is hist.")
        exit(0)

    clf = XGBoostVariant(model_name=args.model_name, train_set_file=args.cluster,
                         method=args.method, objective=args.objective, base_score=args.base_score, grow_policy=args.grow_policy, validation=args.validate,
                         num_trees=args.num_trees, early_stopping=args.early_stopping, max_depth=args.max_depth, min_child_weight=args.min_child_weight, eta=args.eta,
                         sample_bytree=args.sample_bytree, sample_by_level=args.sample_bylevel, sample_bynode=args.sample_bynode,

                         num_parallel_trees=args.num_parallel_trees,

                         data_ensemble_file=args.data_ensemble,
                         features_sets_dir=args.features_sets_dir,
                         random_state=args.random_state,
                         )

    subsampler = None
    if args.subsample_ratios is not None:
        args.subsample_ratios = ast.literal_eval(args.subsample_ratios)

    if args.annotations is not None:
        snp_ids = read_annotations(file_path=args.annotations, note_type=args.annotation_type, n_tissue=args.n_tissue)
        if args.subsample_ratios is not None:
            if args.uniform_over_chromosomes:
                subsampler = lambda x, y: subsample_uniform_chromosomes(subsample_annotated(data=x, snp_ids=snp_ids), subsample_ratio=y)
            else:
                subsampler = lambda x, y: subsample_standard(subsample_annotated(data=x, snp_ids=snp_ids),subsample_ratio=y)
        else:
            subsampler = lambda x: subsample_annotated(data = x, snp_ids=snp_ids)
    elif args.subsample_ratios is not None:
        if args.uniform_over_chromosomes:
            subsampler = subsample_uniform_chromosomes
        else:
            subsampler = subsample_standard
    else:
        subsampler = lambda x: x

    try:
        os.mkdir(args.model_name)
    except FileExistsError:
        logging.info(f"Warning: overwriting existing files in {args.model_name}")
    os.chdir(args.model_name)


    data = clf.read(data_file=args.dataset)

    if args.uniform_over_chromosomes:
        logging.info("Finding chromosomes")
        start_t = time.time()
        chromosomes_list = np.array([name.split(":")[0] for name in data.columns])
        count = np.unique(chromosomes_list, return_counts=True)
        chromosomes_count = {count[0][i] : count[1][i] for i in range(len(count[0]))}
        chromosome_info = (chromosomes_list, chromosomes_count)
        subsampler = lambda x, y: subsample_uniform_chromosomes(data=x, subsample_ratio=y, chromosome_info=chromosome_info)
        logging.info(f"Chromosomes found in {time.time() - start_t : .2f}s")

    to_map = clf.subsample(data=data, subsampling_ratios=args.subsample_ratios, subsampling=subsampler, iterations=args.iterations)

    if args.stdout_values:
        print("subsample_ratio;accuracy;f1;iteration")

    train_wrapper = partial(train, target_file = args.target, stdout = args.stdout_values)

    with ProcessPoolExecutor() as pool:
        pool.map(train_wrapper, to_map, chunksize=1)
