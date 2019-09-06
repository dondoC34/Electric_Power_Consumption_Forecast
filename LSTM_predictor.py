import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from Data_Preprocessor import DataProcessor
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from tqdm import tqdm

class LSTM_predictor:

    def __init__(self, loss_funtion, data_frame, inputs, output, hidden_layers_nr=2, lstm_x_hidden_layer=[90, 90],  a_function="relu",
                 optimizer="adam", steps_back=10, include_current_val=True, filename=""):
        self.frame = data_frame
        self.inputs = inputs
        self.output = output
        self.steps_back = steps_back
        self.hidden_layers_nr = hidden_layers_nr
        self.include_current_val = include_current_val
        self.lstm_x_hidden_layer = lstm_x_hidden_layer
        self.a_function = a_function
        self.optimizer = optimizer
        self.loss_funtion = loss_funtion

        if include_current_val:
            self.input_shape = (len(self.inputs), steps_back + 1)
        else:
            self.input_shape = (len(self.inputs), steps_back)
        if filename:
            self.loaded = True
            self.model = load_model(filepath=filename)
        else:
            self.loaded = False
            self.model = Sequential()
            r_sequence = False
            if hidden_layers_nr > 1:
                r_sequence = True
            self.model.add(LSTM(lstm_x_hidden_layer[0], input_shape=self.input_shape, activation=a_function,
                                return_sequences=r_sequence))
            for i in range(1, hidden_layers_nr):
                r_sequence = False
                if i + 1 < hidden_layers_nr:
                    r_sequence = True
                self.model.add(LSTM(lstm_x_hidden_layer[i], activation=a_function, return_sequences=r_sequence))
            self.model.add(Dense(1, activation=a_function))
            self.model.compile(optimizer=optimizer, loss=loss_funtion)

    def define_inputs_with_back_steps(self):

        result = []
        tmp = []

        for name in self.inputs:
            for i in range(0, self.steps_back):
                tmp.append(str(name) + "_prev_value_" + str(self.steps_back - i))
            if self.include_current_val:
                tmp.append(name)
            result.append(tmp.copy())
            tmp.clear()

        return result

    def prepare_model_input(self, target_index=-1, frame=None):
        result = []
        final_result = []
        column_names = self.define_inputs_with_back_steps()

        if frame is None:
            if target_index < 0:
                for i in range(len(self.frame)):
                    if i < self.steps_back:
                        continue
                    for name in column_names:
                        result.append(list(self.frame[self.steps_back::].loc[i, name]).copy())
                    final_result.append(result.copy())
                    result.clear()
            else:
                for name in column_names:
                    result.append(list(self.frame[self.steps_back::].loc[target_index, name]).copy())
                final_result.append(result.copy())
        else:
            if target_index < 0:
                for i in range(len(frame)):
                    for name in column_names:
                        result.append(list(frame.loc[i, name]).copy())
                    final_result.append(result.copy())
                    result.clear()
            else:
                for name in column_names:
                    result.append(list(frame.loc[target_index, name]).copy())
                final_result.append(result.copy())

        return np.array(final_result)

    def prepare_second_model_input(self, column_names, target_index=-1, frame=None):
        result = []
        final_result = []

        if frame is None:
            if target_index < 0:
                for i in range(len(self.frame)):
                    if i < self.steps_back:
                        continue
                    for name in column_names:
                        result.append(list(self.frame[self.steps_back::].loc[i, name]).copy())
                    final_result.append(result.copy())
                    result.clear()
            else:
                for name in column_names:
                    result.append(list(self.frame[self.steps_back::].loc[target_index, name]).copy())
                final_result.append(result.copy())
        else:
            if target_index < 0:
                for i in range(len(frame)):
                    for name in column_names:
                        result.append(list(frame.loc[i, name]).copy())
                    final_result.append(result.copy())
                    result.clear()
            else:
                for name in column_names:
                    result.append(list(frame.loc[target_index, name]).copy())
                final_result.append(result.copy())

        return np.array(final_result)

    def train_model(self, epochs, batch, verbose=2, validation_split=0.2, callBacks=[]):
        
        input_array = self.prepare_model_input()
        
        if not self.loaded:
            self.model.fit(x=input_array, y=self.frame[self.steps_back::][self.output], epochs=epochs, batch_size=batch,
                           verbose=verbose, validation_split=validation_split, callbacks=callBacks)

    def basic_testing(self, validation_split=0.2, batch=100, verbose=2):
        input_array = self.prepare_model_input()
        if self.include_current_val:
            steps = self.steps_back + 1
        else:
            steps = self.steps_back
        result = self.model.evaluate(x=input_array[int((1 - validation_split) * len(input_array))::],
                                     y=self.frame[int((1 - validation_split) * len(input_array)) + steps - 1::][self.output]
                                     , batch_size=batch, verbose=verbose)
        print(result)
        return result

    def chain_testing(self, chain_steps, column_to_overwirte, mape=False, result_indexes=[], validation_split=0.2):

        chain_predictions = []
        tmp_list = []
        mse_list = []
        mape_list = []
        predictions_result = []
        if self.include_current_val:
            steps = self.steps_back + 1
        else:
            steps = self.steps_back
        initial_index = int((1 - validation_split) * len(self.frame)) + steps - 1

        test_frame = self.frame[initial_index::]
        test_frame.reset_index(inplace=True, drop=True)

        column_names = self.define_inputs_with_back_steps()
        for starting_index in tqdm(range(len(test_frame) - chain_steps)):

            tmp_test_frame = test_frame.copy()
            tmp_test_frame["AvgP"] = tmp_test_frame["AvgP"].astype(np.float)
            for i in range(starting_index, starting_index + chain_steps):
                input = self.prepare_model_input(target_index=i, frame=tmp_test_frame)
                result = self.model.predict(input)
                tmp_test_frame.at[i + 1, column_to_overwirte] = result[0]

                for name in column_names:
                    for k in range(len(name) - 1):
                        tmp_test_frame.at[i + 1, name[k]] = tmp_test_frame.at[i, name[k + 1]]

                chain_predictions.append(result[0])


            MSE = mean_squared_error(test_frame.loc[starting_index:(starting_index + chain_steps - 1), self.output], np.array(chain_predictions))
            mse_list.append(MSE)
            if mape:
                mape = np.mean(np.abs((np.array(list(test_frame.loc[starting_index:(starting_index + chain_steps - 1), self.output])) - np.array(chain_predictions)) / (
                    np.array(list(test_frame.loc[starting_index:(starting_index + chain_steps - 1), self.output]))))) * 100
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

    def double_chain_testing(self, chain_steps, column_to_overwirte, model_2_inputs,
                             model_2_output, column_to_overwrite_2, mape=False, result_indexes=[],
                             validation_split=0.2, filename="", include_current_val_2=True):

        chain_predictions = []
        tmp_list = []
        mse_list = []
        mape_list = []
        predictions_result = []
        model_2_column_names = []

        if self.include_current_val:
            steps = self.steps_back + 1
        else:
            steps = self.steps_back
        initial_index = int((1 - validation_split) * len(self.frame)) + steps - 1

        for name in model_2_inputs:
            for i in range(0, self.steps_back):
                tmp_list.append(str(name) + "_prev_value_" + str(self.steps_back - i))
            if include_current_val_2:
                tmp_list.append(name)
            model_2_column_names.append(tmp_list.copy())
            tmp_list.clear()
        model_2_input_shape = (len(model_2_inputs), steps)

        if filename:
            model_2 = load_model(filepath=filename)
        else:
            model_2 = Sequential()
            model_2.add(LSTM(10, input_shape=model_2_input_shape))
            model_2.add(Dense(1, activation="linear"))
            model_2.compile(optimizer=self.optimizer, loss="mean_squared_error")
            model_2_prepared_input = self.prepare_second_model_input(column_names=model_2_column_names)
            es = EarlyStopping(patience=5)
            model_2.fit(x=model_2_prepared_input, y=self.frame[self.steps_back::][model_2_output],
                    validation_split=0.2, verbose=2, batch_size=10, epochs=100, callbacks=[es])

        test_frame = self.frame[initial_index::]
        test_frame.reset_index(inplace=True, drop=True)
        column_names = self.define_inputs_with_back_steps()

        for starting_index in tqdm(range(len(test_frame) - chain_steps)):

            tmp_test_frame = test_frame.copy()
            tmp_test_frame["AvgP"] = tmp_test_frame["AvgP"].astype(np.float)

            for i in range(starting_index, starting_index + chain_steps):
                input = self.prepare_model_input(target_index=i, frame=tmp_test_frame)
                result = self.model.predict(input)
                tmp_test_frame.at[i + 1, column_to_overwirte] = result[0]
                chain_predictions.append(result[0])

                input = self.prepare_second_model_input(column_names=model_2_column_names, target_index=i,
                                                        frame=tmp_test_frame)

                result = model_2.predict(x=input)
                tmp_test_frame.at[i + 1, column_to_overwrite_2] = result[0]

                for name in column_names:
                    for k in range(len(name) - 1):
                        tmp_test_frame.at[i + 1, name[k]] = tmp_test_frame.at[i, name[k + 1]]

            MSE = mean_squared_error(test_frame.loc[starting_index:(starting_index + chain_steps - 1), self.output], np.array(chain_predictions))
            mse_list.append(MSE)
            if mape:
                mape = np.mean(np.abs((np.array(list(test_frame.loc[starting_index:(starting_index + chain_steps - 1), self.output])) - np.array(chain_predictions)) / (
                    np.array(list(test_frame.loc[starting_index:(starting_index + chain_steps - 1), self.output]))))) * 100
                mape_list.append(mape)
            if (starting_index in result_indexes) or (-1 in result_indexes):
                predictions_result.append(chain_predictions.copy())

            chain_predictions.clear()

        mse_result = np.sqrt(np.mean(mse_list))
        print("MSE: " + str(mse_result))
        if mape:
            mape_result = np.mean(mape_list)
            print("MAPE: " + str(mape_result))
            return mse_result, mape_result
        else:
            return mse_result


if __name__ == "__main__":

    pred = LSTM_predictor("mean_absolute_percentage_error", read_csv("AVGpower_1_floor.csv"), ["AvgT", "Hour", "AvgP"],
                          "AvgP_next_value_1", filename="", steps_back=10, hidden_layers_nr=4,
                          lstm_x_hidden_layer=[5, 5, 5, 5])
    es = EarlyStopping(patience=4)
    pred.train_model(epochs=500, batch=20, callBacks=[])

    #file = open("Chain_LSTM_floor1.txt", "w")
    #
    #for i in range(1, 16):
    #    print("iteration: " + str(i))
    #    a, b = pred.double_chain_testing(chain_steps=i, column_to_overwirte="AvgP", column_to_overwrite_2="AvgT",
    #                                 model_2_inputs=["AvgT", "Hour"], model_2_output="AvgT_next_value_1", mape=True,
    #                                 filename="")
    #    file.write(str(b))
    #    file.write(",")
    #
    #file.close()
    ## pred.basic_testing()



