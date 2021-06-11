import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# PREDICTION OF GLOBAL HORIZONTAL IRRADIANCE using NSRDB Dataset


class GLM:
    # Constructor
    def __init__(self):
        self.dataframe = pd.read_excel('E:/myCODING/Machine Learning/Solar_Radiation/NSRDB_Ghi.xlsx', engine='openpyxl')

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
        print("\nPREDICTION OF GLOBAL HORIZONTAL IRRADIANCE: ")
        # DATA
        print("Initializing. . . ")
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(self.X.reshape(-1, 4))
        y = sc_y.fit_transform(self.y.reshape(-1, 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Model (Assumed Normally Distributed, 75 Epochs)
        from sklearn.linear_model import TweedieRegressor
        glm_model = TweedieRegressor(max_iter=75, power=0, alpha=0.5, link='identity')
        glm_model_fit = glm_model.fit(X_train.reshape(-1, 4), y_train.reshape(-1, 1))
        y_preds = glm_model.predict(X_test)
        y_pred = sc_y.inverse_transform(y_preds)

        # Model Evaluation
        # Mean Squared Error
        print("• MSE:", mean_squared_error(y_test, y_preds))
        # Root Mean Squared Error
        print("• RMSE:", mean_squared_error(y_test, y_preds, squared=False))
        # Mean Absolute Error
        print("• MAE:", mean_absolute_error(y_test, y_preds))
        print("\nGeneralization Progress: ")
        df = pd.DataFrame({'Real Values': sc_y.inverse_transform(y_test.reshape(-1)), 'Predicted Values': y_pred})
        print(df)


glm = GLM()
glm.dataset_inspection()
glm.data_set_distribution("label")
glm.model()
