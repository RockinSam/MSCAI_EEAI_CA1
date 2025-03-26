import random, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from Config import *

seed = 0
random.seed(seed)
np.random.seed(seed)

class Data():

    def __init__(self, X: np.ndarray, df) -> None:

        for typ in Config.TYPE_COLS:
            le = LabelEncoder()
            df[typ] = le.fit_transform(df[typ].astype(str))  
        y = df[Config.TYPE_COLS].values.astype(int) 

        y_series = pd.DataFrame(y) 
        
        good_y_value = y_series.apply(pd.Series.value_counts).sum(axis=1)
        good_y_value = good_y_value[good_y_value >= 3].index

        if len(good_y_value) < 1:
            print("None of the labels have more than 3 records: Skipping ...")
            self.X_train = None
            return
        
        mask = y_series.apply(lambda row: any(row.isin(good_y_value)), axis=1)
        y_good = y[mask]
        X_good = X[mask]

        new_test_size = X.shape[0] * 0.2 / X_good.shape[0]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_good, y_good, test_size=new_test_size, random_state=0
        )
        
        self.y = y
        self.embeddings = X
        self.classes = good_y_value


    def get_type(self):
        return self.y
    
    def get_X_train(self):
        return self.X_train
    
    def get_X_test(self):
        return self.X_test
    
    def get_type_y_train(self):
        return self.y_train
    
    def get_type_y_test(self):
        return self.y_test
    
    def get_embeddings(self):
        return self.embeddings
    
    def get_X_DL_test(self):
        return self.X_test
    
    def get_X_DL_train(self):
        return self.X_train
