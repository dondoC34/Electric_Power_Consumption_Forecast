import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from Data_Preprocessor import *
from joblib import load, dump
from tqdm import tqdm


class Linear_predictor:

    def __init__(self, train_frame, test_frame):
        self.test_frame = test_frame
        self.train_frame = train_frame
        self.model = linear_model.Lasso(alpha=0.5, max_iter=2000)
        self.inputs = []
        self.output = []

    def train_model(self, inputs, output, poly_degree=2, filename=""):
        self.inputs = inputs.copy()
        self.output = output
        if filename:
            self.model = load(filename)
        else:
            polynomial_features = PolynomialFeatures(degree=poly_degree)
            x_poly = polynomial_features.fit_transform(self.train_frame[self.inputs])
            self.model.fit(X=x_poly, y=self.train_frame[self.output])

    def save_model(self, filename):
        dump(self.model, filename)

    def basic_testing(self, poly_degree=2, mape=False, verbose=False):
        polynomial_features = PolynomialFeatures(degree=poly_degree)
        x_poly = polynomial_features.fit_transform(self.test_frame[self.inputs])
        result = self.model.predict(x_poly)

        if verbose:
            for i in range(len(result)):
                print("Predicted: " + str(result[i]) + "   Real: " + str(self.test_frame[self.output][i]))

        MSE = np.sqrt(mean_squared_error(self.test_frame[self.output], np.array(result)))
        print("MSE: " + str(MSE))
        if mape:
            mape = np.mean(np.abs((np.array(list(self.test_frame["AvgP_next_value_1"])) - np.array(result)) / (
                np.array(list(self.test_frame["AvgP_next_value_1"]))))) * 100
            print("MAPE: " + str(mape))
            return MSE, mape
        else:
            return MSE

    def chain_testing(self, chain_steps, column_to_overwirte, poly_degree=2, mape=False):

        chain_predictions = []
        tmp_list = []
        mse_list = []
        mape_list = []

        for starting_index in tqdm(range(len(self.test_frame) - chain_steps)):

            tmp_test_frame = self.test_frame.copy()
            # tmp_test_frame["AvgP"] = tmp_test_frame["AvgP"].astype(np.float)
            for i in range(starting_index, starting_index + chain_steps):
                polynomial_features = PolynomialFeatures(degree=poly_degree)
                tmp_list.append(tmp_test_frame.loc[i, self.inputs])
                x_poly = polynomial_features.fit_transform(tmp_list)
                result = self.model.predict(x_poly)
                tmp_list.clear()
                tmp_test_frame.at[i + 1, column_to_overwirte] = result[0]
                chain_predictions.append(result[0])

            MSE = mean_squared_error(self.test_frame.loc[starting_index:(starting_index + chain_steps - 1), self.output], np.array(chain_predictions))
            mse_list.append(MSE)
            if mape:
                mape = np.mean(np.abs((np.array(list(self.test_frame.loc[starting_index:(starting_index + chain_steps - 1), self.output])) - np.array(chain_predictions)) / (
                    np.array(list(self.test_frame.loc[starting_index:(starting_index + chain_steps - 1), self.output]))))) * 100
                mape_list.append(mape)
            chain_predictions.clear()

        mse_result = np.sqrt(np.mean(mse_list))

        if mape:
            mape_result = np.mean(mape_list)
            return mse_result, mape_result
        else:
            return mse_result

    def double_chain_testing(self, chain_steps, column_to_overwrite_1, column_to_overwrite_2, model_2_inputs,
                             model_2_output, poly_degree_1=2, poly_degree_2=2, mape=False, filename_2=""):

        tmp_list = []
        chain_predictions = []
        mse_list = []
        mape_list = []

        if filename_2:
            model_2 = load(filename=filename_2)
        else:
            model_2 = linear_model.LinearRegression()
            polinomial_features = PolynomialFeatures(degree=poly_degree_2)
            x_poly = polinomial_features.fit_transform(self.train_frame[model_2_inputs])
            model_2.fit(X=x_poly, y=self.train_frame[model_2_output])

        for starting_index in tqdm(range(len(self.test_frame) - chain_steps)):

            tmp_test_frame = self.test_frame.copy()
            tmp_test_frame["AvgP"] = tmp_test_frame["AvgP"].astype(np.float)

            for i in range(starting_index, starting_index + chain_steps):

                polynomial_features = PolynomialFeatures(degree=poly_degree_1)
                tmp_list.append(tmp_test_frame.loc[i, self.inputs])
                x_poly = polynomial_features.fit_transform(tmp_list)
                result = self.model.predict(x_poly)
                tmp_list.clear()
                tmp_test_frame.at[i + 1, column_to_overwrite_1] = result[0]
                chain_predictions.append(result[0])

                polynomial_features = PolynomialFeatures(degree=poly_degree_2)
                tmp_list.append(tmp_test_frame.loc[i, model_2_inputs])
                x_poly = polynomial_features.fit_transform(tmp_list)
                result = model_2.predict(x_poly)
                tmp_list.clear()
                tmp_test_frame.at[i + 1, column_to_overwrite_2] = result[0]

            MSE = mean_squared_error(self.test_frame.loc[starting_index:(starting_index + chain_steps - 1), self.output],
                                     np.array(chain_predictions))
            mse_list.append(MSE)
            if mape:
                mape = np.mean(np.abs((np.array(list(
                    self.test_frame.loc[starting_index:(starting_index + chain_steps - 1), self.output])) - np.array(
                    chain_predictions)) / (
                                          np.array(list(
                                              self.test_frame.loc[starting_index:(starting_index + chain_steps - 1),
                                              self.output]))))) * 100
                mape_list.append(mape)
            chain_predictions.clear()

        mse_result = np.sqrt(np.mean(mse_list))
        print("MSE: " + str(mse_result))
        if mape:
            mape_result = np.mean(mape_list)
            print("MAPE: " + str(mape_result))
            return mse_result, mape_result
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
        predictor = Linear_predictor(train_frame, test_frame)
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
        return np.mean(mse_list), np.mean(mape_list)
    else:
        return np.mean(mse_list)


if __name__ == '__main__':

    frame = pd.read_csv("AVGpower_3_floor.csv")
    dp = DataProcessor(frame)
    sets = dp.compute_train_test_set([80, 20])
    train_frame = sets[0]
    test_frame = sets[1]
    predictor = Linear_predictor(train_frame, test_frame)
    predictor.train_model(filename="", poly_degree=4, inputs=["Hour", "AvgT", "AvgP", "Month % 1", "Month % 2", "Month % 3", "Month % 4", "Month % 5", "Month % 6", "Month % 7", "Month % 8", "Month % 9", "Month % 10", "Month % 11", "Month % 12"], output="AvgP_next_value_1")
    predictor.basic_testing(mape=True, poly_degree=4)
    # file = open("Chain_Linear_2_Lasso_floor1.txt", "w")
    # for i in range(1, 16):
    #     print("iteration: " + str(i))
    #     a, b = predictor.double_chain_testing(chain_steps=i, column_to_overwrite_1="AvgP", column_to_overwrite_2="AvgT",
    #                                    model_2_inputs=["Day", "Hour", "Minutes", "Month % 1",
    #                                    "Month % 2", "Month % 3", "Month % 4", "Month % 5", "Month % 6", "Month % 7",
    #                                    "Month % 8", "Month % 9", "Month % 10", "Month % 11", "Month % 12", "AvgT"],
    #                                   model_2_output="AvgT_next_value_1", mape=True, poly_degree_1=1, poly_degree_2=1,
    #                                           filename_2="")
    #     file.write(str(b))
    #     file.write(",")
    #
    # file.close()
