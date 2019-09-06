import pandas as pd
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from Data_Preprocessor import *
from tqdm import tqdm
from joblib import dump, load


class GBoost_regression_trees_predictor:

    def __init__(self, train_frame, test_frame, verbose_model=1, num_of_trees=15000):
        self.test_frame = test_frame
        self.train_frame = train_frame
        self.num_of_trees = num_of_trees
        self.model = ensemble.GradientBoostingRegressor(n_estimators=num_of_trees, max_depth=8, max_features="auto",
                                                        verbose=verbose_model, max_leaf_nodes=32, learning_rate=0.1)
        self.inputs = []
        self.output = []

    def train_model(self, inputs, output, filename=""):
        self.inputs = inputs.copy()
        self.output = output
        if filename:
            self.model = load(filename)
        else:
            self.model.fit(X=self.train_frame[self.inputs], y=self.train_frame[self.output])

    def save_model(self, filename):
        dump(self.model, filename)

    def basic_testing(self, mape=False, verbose=False):

        result = self.model.predict(self.test_frame[self.inputs])

        if verbose:
            for i in range(len(result)):
                print("Predicted: " + str(result[i]) + "   Real: " + str(self.test_frame[self.output][i]))

        MSE = np.sqrt(mean_squared_error(self.test_frame[self.output], np.array(result)))
        print("MSE: " + str(MSE))
        if mape:
            mape = np.mean(np.abs((np.array(list(self.test_frame[self.output])) - np.array(result)) / (
                np.array(list(self.test_frame[self.output]))))) * 100
            print("MAPE: " + str(mape))
            return MSE, mape
        else:
            return MSE

    def chain_testing(self, chain_steps, column_to_overwirte, mape=False, result_indexes=[]):

        chain_predictions = []
        tmp_list = []
        mse_list = []
        mape_list = []
        predictions_result = []

        for starting_index in tqdm(range(len(self.test_frame) - chain_steps)):

            tmp_test_frame = self.test_frame.copy()
            tmp_test_frame["AvgP"] = tmp_test_frame["AvgP"].astype(np.float)
            for i in range(starting_index, starting_index + chain_steps):
                tmp_list.append(tmp_test_frame.loc[i, self.inputs])
                result = self.model.predict(tmp_list)
                tmp_list.clear()
                tmp_test_frame.at[i + 1, column_to_overwirte] = result[0]
                chain_predictions.append(result[0])

            MSE = mean_squared_error(self.test_frame.loc[starting_index:(starting_index + chain_steps - 1), self.output], np.array(chain_predictions))
            mse_list.append(MSE)
            if mape:
                mape = np.mean(np.abs((np.array(list(self.test_frame.loc[starting_index:(starting_index + chain_steps - 1), self.output])) - np.array(chain_predictions)) / (
                    np.array(list(self.test_frame.loc[starting_index:(starting_index + chain_steps - 1), self.output]))))) * 100
                mape_list.append(mape)
            if (starting_index in result_indexes) or (-1 in result_indexes):
                predictions_result.append(chain_predictions.copy())

            chain_predictions.clear()

        mse_result = np.sqrt(np.mean(mse_list))
        print("MSE: " + str(mse_result))
        if mape:
            mape_result = np.mean(mape_list)
            print("MAPE: " + str(mape_result))
            return mse_result, mape_result, predictions_result
        else:
            return mse_result, predictions_result

    def double_chain_testing(self, chain_steps, column_to_overwrite_1, column_to_overwrite_2, model_2_inputs,
                             model_2_output, result_indexes=[], mape=False, filename_2=""):

        tmp_list = []
        chain_predictions = []
        mse_list = []
        mape_list = []
        predictions_result = []
        if filename_2:
            model_2 = load(filename_2)
        else:
            model_2 = ensemble.GradientBoostingRegressor(n_estimators=self.num_of_trees, max_depth=8, max_features="auto",
                                                        verbose=0, max_leaf_nodes=32)
            model_2.fit(X=self.train_frame[model_2_inputs], y=self.train_frame[model_2_output])

        for starting_index in tqdm(range(len(self.test_frame) - chain_steps)):

            tmp_test_frame = self.test_frame.copy()
            tmp_test_frame["AvgP"] = tmp_test_frame["AvgP"].astype(np.float)
            for i in range(starting_index, starting_index + chain_steps):

                tmp_list.append(tmp_test_frame.loc[i, self.inputs])
                result = self.model.predict(tmp_list)
                tmp_list.clear()
                tmp_test_frame.at[i + 1, column_to_overwrite_1] = result[0]
                chain_predictions.append(result[0])

                tmp_list.append(tmp_test_frame.loc[i, model_2_inputs])
                result = model_2.predict(tmp_list)
                tmp_list.clear()
                tmp_test_frame.at[i + 1, column_to_overwrite_2] = result[0]

            mse = mean_squared_error(self.test_frame.loc[starting_index:(starting_index + chain_steps - 1), self.output],
                                     np.array(chain_predictions))

            mse_list.append(mse)
            if mape:
                mape = np.mean(np.abs((np.array(list(
                    self.test_frame.loc[starting_index:(starting_index + chain_steps - 1), self.output])) - np.array(
                    chain_predictions)) / (
                                          np.array(list(
                                              self.test_frame.loc[starting_index:(starting_index + chain_steps - 1),
                                              self.output]))))) * 100
                mape_list.append(mape)

            if (starting_index in result_indexes) or (-1 in result_indexes):
                predictions_result.append(chain_predictions.copy())
            chain_predictions.clear()

        mse_result = np.sqrt(np.mean(mse_list))
        print("MSE: " + str(mse_result))
        if mape:
            mape_result = np.mean(mape_list)
            print("MAPE: " + str(mape_result))
            if len(predictions_result) > 0:
                return mse_result, mape_result, predictions_result
            else:
                return mse_result, mape_result
        else:
            if len(predictions_result) > 0:
                return mse_result, predictions_result
            else:
                return mse_result

def k_fold_Xvalidation(filename, k, mape=False):

    percentages = [100 / k] * k
    mse_list = []
    mape_list = []

    for test_set_index in tqdm(range(len(percentages))):

        frame = pd.read_csv(filename)
        dp = DataProcessor(frame)
        sets = dp.compute_train_test_set(percentages=percentages, index=test_set_index)
        train_frame = sets[0]
        test_frame = sets[1]
        predictor = GBoost_regression_trees_predictor(train_frame, test_frame, verbose_model=0)
        predictor.train_model(inputs=["Hour", "AvgT", "AvgP", "Month % 1", "Month % 2", "Month % 3", "Month % 4", "Month % 5", "Month % 6", "Month % 7", "Month % 8", "Month % 9", "Month % 10", "Month % 11", "Month % 12"], output="AvgP_next_value_1")
        mse, mape = predictor.double_chain_testing(chain_steps=10, column_to_overwrite_1="AvgP", column_to_overwrite_2="AvgT",
                                        model_2_inputs=["Day", "Hour", "Minutes", "Month % 1",
                                        "Month % 2", "Month % 3", "Month % 4", "Month % 5", "Month % 6", "Month % 7",
                                        "Month % 8", "Month % 9", "Month % 10", "Month % 11", "Month % 12", "AvgT"],
                                        model_2_output="AvgT_next_value_1", mape=True)

        mse_list.append(mse)
        if mape:
            mape_list.append(mape)

    if mape:
        return mse_list, mape_list
    else:
        return mse_list


if __name__ == "__main__":

    frame = pd.read_csv("data-set/AVGpower_3_floor.csv")
    dp = DataProcessor(frame)
    sets = dp.compute_train_test_set([80, 20])
    train_frame = sets[0]
    test_frame = sets[1]
    predictor = GBoost_regression_trees_predictor(train_frame, test_frame, verbose_model=1, num_of_trees=50)
    predictor.train_model(filename="saved-models/GBoost100.joblib",
                          inputs=["Hour", "AvgT", "AvgP", "Month % 1", "Month % 2", "Month % 3", "Month % 4",
                                  "Month % 5", "Month % 6", "Month % 7", "Month % 8", "Month % 9",
                                  "Month % 10", "Month % 11", "Month % 12"], output="AvgP_next_value_1")
    predictor.basic_testing(mape=True)




