import numpy as np
from Config import *
from model.base import BaseModel
from modelling.data_model import Data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class ChainedMultiOutput(BaseModel):


    def __init__(self, name, data: Data):

        self.data = data
        self.model_name = name

        self.model_y2 = RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced')
        self.model_y3 = RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced')
        self.model_y4 = RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced')

        self.y2_pred = None
        self.y3_pred = None
        self.y4_pred = None
    

    def train(self, data):

        self.model_y2.fit(data.X_train, data.y_train[:, 0])
        y2_train_pred = self.model_y2.predict(data.X_train)
        
        X_train_y3 = np.column_stack((data.X_train, y2_train_pred))
        self.model_y3.fit(X_train_y3, data.y_train[:, 1])
        y3_train_pred = self.model_y3.predict(X_train_y3)
        
        X_train_y4 = np.column_stack((data.X_train, y2_train_pred, y3_train_pred))
        self.model_y4.fit(X_train_y4, data.y_train[:, 2])
        

    def predict(self, X_test):
        self.y2_pred = self.model_y2.predict(X_test)
        self.y3_pred = self.model_y3.predict(np.column_stack((X_test, self.y2_pred)))
        self.y4_pred = self.model_y4.predict(np.column_stack((X_test, self.y2_pred, self.y3_pred)))
    

    def print_results(self, data):

        scores = []
        for i in range(len(data.y_test)):
            score = 0
            if self.y2_pred[i] == data.y_test[i][0]:
                score = 33
                if self.y3_pred[i] == data.y_test[i][1]:
                    score = 67
                    if self.y4_pred[i] == data.y_test[i][2]:
                        score = 100
            scores.append(score)
        final_accuracy = np.mean(scores)
        print(f"Evaluation Final Accuracy: {final_accuracy:.2f}%\n")

        
        # print("\nClassification Report for Type2:")
        # print(classification_report(data.y_test[:, 0], self.y2_pred, zero_division=0))
        
        # print("\nClassification Report for Type 2 + Type3:")
        # print(classification_report(data.y_test[:, 1], self.y3_pred, zero_division=0))
        
        # print("\nClassification Report for Type 2 + Type3 + Type4:")
        # print(classification_report(data.y_test[:, 2], self.y4_pred, zero_division=0))


    def data_transform(self):
        return super().data_transform()
