import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from utils import *

def get_model(model_name='random_forest', **kwargs):
    if model_name == 'random_forest':
        return RandomForestRegressor(**kwargs)
    elif model_name == 'adaboost':
        return AdaBoostRegressor(**kwargs)
    else:
        raise ValueError("지원하지 않는 모델입니다: 'random_forest' 또는 'adaboost'를 선택하세요.")

def train_and_eval(model,dataset, label,Category_variables,Numeric_variables,Target_variables
):
    print(f"==== Training on {label} dataset ====")

    train_df, vald_df, test_df = dataset_split(dataset)
    X_train = train_df[list(Category_variables)+Numeric_variables]
    y_train = train_df[Target_variables]
    

    model.fit(X_train, y_train)
    
    x_test = test_df[list(Category_variables)+Numeric_variables]
    test_predict = model.predict(x_test)
    
    mse = mean_squared_error(test_predict,test_df[Target_variables])
    rmse = np.sqrt(mean_squared_error(test_predict,test_df[Target_variables]))
    r2 = r2_score(test_predict,test_df[Target_variables])
           
    print("MSE : ", mse)
    print("RMSE : ",rmse)
    print("R_score : ",r2)
    
    return {"MSE" : mse, "RMSE" : rmse, "R_score" : r2}