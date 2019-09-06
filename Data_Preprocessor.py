import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class DataProcessor:

    def __init__(self, frame):
        pd.set_option('display.max_rows', 3000)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        pd.options.mode.chained_assignment = None
        self.frame = frame

    def column_to_one_hot(self, column_name, possibilities):
        num_of_columns = len(possibilities)

        for i in range(num_of_columns):
            self.frame[column_name + " % " + str(possibilities[i])] = self.frame[column_name].apply(lambda x: 1 if x == possibilities[i] else 0)



    def remove_negative_power(self, final_value):
        self.frame["AvgP"] = self.frame["AvgP"].apply(lambda x: final_value if x < 0 else x)

    def write_csv(self, name):
        self.frame.to_csv(name, index=False)

    def concat(self, frame):
        self.frame = pd.concat([self.frame, frame])
        self.frame.reset_index(drop=True, inplace=True)
        self.frame.sort_values(["Year", "Month", "Day", "Hour", "Minutes"], ascending=[True] * 5, inplace=True)

    def print_frame(self, head=0):
        if head:
            pd.set_option('display.max_rows', head)
            print(self.frame.head(head))
        else:
            print(self.frame)

    def interpolate(self, column_name, treshold_value):
        low_index = 0
        high_index = 0
        low_value = 0
        high_value = 0
        streak = False
        self.frame.reset_index(inplace=True, drop=True)

        for i, row in self.frame.iterrows():

            if (row[column_name] >= treshold_value) and not streak:
                low_value = row[column_name]

            elif row[column_name] < treshold_value and not streak:
                streak = True
                low_index = i

            elif row[column_name] >= treshold_value and streak:
                streak = False
                high_index = i - 1
                high_value = row[column_name]

                for k in range(low_index, high_index + 1):
                    interpolated_value = (k - low_index + 1) / (high_index - low_index + 2) * (high_value - low_value) + low_value
                    self.frame.at[k, column_name] = interpolated_value
                low_value = high_value

    def compute_train_test_set(self, percentages, index=-1, columns_from_train_set=[]):
        data_len = len(self.frame)
        result = []
        slices = []
        prev_value = 0
        following_val = 0

        for percentage in percentages:
            following_val = prev_value + int(data_len * percentage / 100)
            slices.append(self.frame[prev_value:following_val])
            prev_value = following_val

        test_set = slices.pop(index)
        train_set = pd.concat(slices)
        test_set.reset_index(inplace=True, drop=True)

        for i, row in test_set.iterrows():
            for column_name in columns_from_train_set:
                values = set(train_set.loc[(train_set["Day"] == row["Day"]) & (train_set["Month"] == row["Month"]), column_name])
                if len(values) == 0:
                    values = set(
                        train_set.loc[(train_set["Day"] == (row["Day"] - 1)) & (train_set["Month"] == row["Month"]), column_name])

                if len(values) == 0:
                    print("exception")
                test_set.at[i, column_name] = sum(values) / len(values)

        train_set.reset_index(drop=True, inplace=True)

        return train_set, test_set

    def normalize_columns(self, columns):

        for column in columns:
            self.frame[columns] = self.frame[columns] / self.frame[columns].max()

    def add_next_values(self, column_name, steps_ahead=1):
        self.frame.reset_index(drop=True, inplace=True)
        self.frame[column_name + "_next_value_" + str(steps_ahead)] = 0.0
        tmp = steps_ahead
        while tmp > 0:
            self.frame.at[len(self.frame) - tmp, column_name + "_next_value_" + str(steps_ahead)] = self.frame.at[len(self.frame) - 1, column_name]
            tmp -= 1

        for i, row in self.frame.iterrows():
            if i < steps_ahead:
                continue
            self.frame.at[i - steps_ahead, column_name + "_next_value_" + str(steps_ahead)] = row[column_name]

    def add_previous_values(self, column_name, steps_back=1):
        self.frame.reset_index(drop=True, inplace=True)
        self.frame[column_name + "_prev_value_" + str(steps_back)] = 0.0

        for i, row in self.frame.iterrows():
            if i + steps_back + 1 > len(self.frame):
                break
            self.frame.at[i + steps_back, column_name + "_prev_value_" + str(steps_back)] = row[column_name]

        tmp = steps_back - 1

        while tmp >= 0:
                self.frame.at[tmp, column_name + "_prev_value_" + str(steps_back)] = self.frame.at[0, column_name]
                tmp -= 1

    def add_daily_avg_value(self, column_name):
        day = 0
        values = []
        self.frame[column_name + "_daily_avg_val"] = 0.0

        day = self.frame.at[0, "Day"]
        for i, row in self.frame.iterrows():

            if i == len(self.frame) - 1:
                index = len(values) + 1
                mean = sum(values) / len(values)
                values.clear()
                values.append(self.frame.at[i, column_name])

                while index > 0:
                    self.frame.at[i + 1 - index, column_name + "_daily_avg_val"] = mean
                    index -= 1

            if self.frame.at[i, "Day"] == day:
                values.append(self.frame.at[i, column_name])
            else:
                day = self.frame.at[i, "Day"]
                index = len(values)
                mean = sum(values) / len(values)
                values.clear()
                values.append(self.frame.at[i, column_name])

                while index > 0:
                    self.frame.at[i - index, column_name + "_daily_avg_val"] = mean
                    index -= 1

    def add_mean_variance(self, column_name, window):
        current_mean = 0
        data = []
        self.frame.reset_index(inplace=True, drop=True)

        self.frame[column_name + "_mean"] = 0.0
        self.frame[column_name + "_var"] = 0.0

        for i, row in self.frame.iterrows():

            if len(data) == window:
                data.__delitem__(0)

            data.append(row[column_name])

            current_mean = sum(data) / len(data)
            current_var = 0.0
            for datum in data:
                current_var = current_var + (datum - current_mean) ** 2
                current_var = 1 / len(data) * current_var
                current_var = np.sqrt(current_var)
            self.frame.at[i, column_name + "_mean"] = current_mean
            self.frame.at[i, column_name + "_var"] = current_var

    def add_days_of_the_week(self):

        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        index = 0
        previous_day = self.frame.at[0, "Day"]

        self.frame["Day_of_the_week"] = "Unknown"

        for i, row in self.frame.iterrows():

            current_day = self.frame.at[i, "Day"]

            if current_day < previous_day:
                index = index + current_day
            else:
                index = index + (current_day - previous_day)

            while index > 6:
                index = index - 7

            self.frame.at[i, "Day_of_the_week"] = days[index]
            previous_day = current_day

    def find_missing_samples(self):
        result_indexes = []
        minutes = [0, 15, 30, 45]
        hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

        minutes_index = 0
        hours_index = 17

        for i, row in self.frame.iterrows():

            while (row["Hour"] != hours[hours_index]) or (row["Minutes"] != minutes[minutes_index]):
                result_indexes.append(1)
                minutes_index += 1
                while minutes_index > 3:
                    minutes_index -= 4
                    hours_index += 1
                    while hours_index > 23:
                        hours_index -= 24
            result_indexes.append(0)
            minutes_index += 1
            while minutes_index > 3:
                minutes_index -= 4
                hours_index += 1
                while hours_index > 23:
                    hours_index -= 24
        return result_indexes

    def adjust_next_value(self, column_name):
        minutes = [0, 15, 30, 45]
        hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        gap_counter = 0

        minutes_index = self.frame.at[0, "Minutes"]
        hours_index = self.frame.at[0, "Hour"]

        for i, row in self.frame.iterrows():
            if i == len(self.frame) - 1:
                break
            minutes_index += 1
            while minutes_index > 3:
                minutes_index -= 4
                hours_index += 1
                while hours_index > 23:
                    hours_index -= 24

            while (self.frame.at[i + 1, "Minutes"] != minutes[minutes_index]) or (self.frame.at[i + 1, "Hour"] != hours[hours_index]):
                gap_counter += 1
                minutes_index += 1
                while minutes_index > 3:
                    minutes_index -= 4
                    hours_index += 1
                    while hours_index > 23:
                        hours_index -= 24

            extrapolation_coeff = 1 / (1 + np.exp(- (gap_counter - 1) / 10))
            interpolation_coeff = 1 / (1 + np.exp((gap_counter - 1) / 10))

            if gap_counter == 0:
                continue
            gap_counter = 0

            y_1 = self.frame.at[i - 1, column_name]
            y_2 = self.frame.at[i - 2, column_name]
            x_1 = i - 1
            x_2 = i - 2

            extrapolation_value = y_1 + (y_2 - y_1) * (i - x_1) / (x_2 - x_1)

            y_1 = self.frame.at[i - 1, column_name]
            y_2 = self.frame.at[i + 1, column_name]
            x_1 = i - 1
            x_2 = i + 1

            interpolation_value = y_1 + (y_2 - y_1) * (i - x_1) / (x_2 - x_1)

            self.frame.at[i, column_name] = interpolation_coeff * interpolation_value + extrapolation_coeff * extrapolation_value

    def remove_duplicated_samples(self):
        deletion_indexes = []
        for i, row in self.frame.iterrows():
            if i == len(self.frame) - 1:
                break
            if (self.frame.at[i + 1, "Minutes"] == row["Minutes"]) and (self.frame.at[i + 1, "Hour"] == row["Hour"]):
                deletion_indexes.append(i)

        self.frame = self.frame.drop(deletion_indexes, axis=0)


if __name__ == '__main__':
    result = []
    frame = pd.read_csv("data-set/AVGpower_1_floor.csv")

    dp = DataProcessor(frame)
    dp.add_days_of_the_week()
    sns.boxplot(x="Day_of_the_week", y="AvgP", data=dp.frame)
    plt.title("Floor 1")
    plt.show()








