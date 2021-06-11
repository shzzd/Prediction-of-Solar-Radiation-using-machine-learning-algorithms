import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# PREDICTION OF GLOBAL HORIZONTAL IRRADIANCE using NSRDB Dataset


class SvR:
    def __init__(self):
        # Extract Data
        self.dataframe = pd.read_excel('C:/Users/USER/PycharmProjects/Neural_Network/MachineLearning/Solar Irradiance/NSRDB_Ghi.xlsx')

    def dataset_inspection(self):
        # DATA OPERATION
        self.dataset = self.dataframe.iloc[:, 2:7]
        self.dataset = self.dataset.apply(pd.to_numeric, errors='coerce')
        self.dataset = self.dataset.dropna()
        self.dataset = self.dataset.reset_index(drop=True)

        print("Displaying first 5 rows of our data set: \n", self.dataset.head())
        # All but 1st Column
        self.X = self.dataset.iloc[:, 1:].values
        # 1st Column
        self.y = self.dataset.iloc[:, :1].values
        self.y = np.array(self.y).reshape(-1, 1)
        self.radiation = self.dataframe['GHI (W/m^2)']

    def data_set_distribution(self, input):
        self.y_label = self.radiation.values
        title = 'Histogram of Radiation'
        xlabel = 'Global Horizontal Irradiance'

        # Plot
        plt.hist(self.y_label, bins=10)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.show()

    def model(self):
        print("\nProcess Initializing. . .")
        # PARTITIONS PROCESS
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(self.X.reshape(-1, 4))
        y = sc_y.fit_transform(self.y.reshape(-1, 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # MODEL
        regressor = SVR(kernel='rbf')
        svr_fit = regressor.fit(X_train.reshape(-1, 4), y_train.reshape(-1, 1))
        y_preds = regressor.predict(X_test)
        y_pred = sc_y.inverse_transform(y_preds)
        n, p = np.shape(X_test)
        print("Number of Data n= ", n)
        print("Number of Predictors p= ", p)

        # EVALUATING METRICS
        # R-Squared
        score = svr_fit.score(X_test, y_test)
        Adjusted_r2 = 1 - (1 - (score * score)) * (n - 1) / (n - p - 1)
        print("\n• R-Squared: ", score)
        print("• Adjusted R-Squared: ", Adjusted_r2)
        # Mean Squared Error
        print("• MSE:", mean_squared_error(y_test, y_preds, squared=True))
        # Root Mean Squared Error
        print("• RMSE:", mean_squared_error(y_test, y_preds, squared=False))
        # Mean Absolute Error
        print("• MAE:", mean_absolute_error(y_test, y_preds))
        print("\nGeneralization Progress: ")
        df = pd.DataFrame({'Real Values': sc_y.inverse_transform(y_test.reshape(-1)), 'Predicted Values': y_pred})
        print(df)


svr = SvR()
svr.dataset_inspection()
svr.data_set_distribution('label')
svr.model()
