import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def data_processing():
    dir_paht = "./data/PART_I/file.csv"
    
    df = pd.read_csv(dir_paht)
    
    # Variable 정리
    Category_variables = ['Category1', 'Category2', 'Category3', 'Category4', 'Category5', 'Category6', 'Category7', 'Category8']
    Numeric_variables = ['Chemical1', 'Chemical2', 'Chemical3', 'Chemical4', 'Chemical5','Chemical6', 'Chemical7', 'Chemical8', 'Chemical9', 'Chemical10', 'Chemical11', 'Chemical12', 'Chemical13', 'Chemical14',
                        'Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Var6', 'Var7', 'Var8', 'Var9', 'Var10', 'Var11', 'Var12', 'Var13',
                        'Var14', 'Var15', 'Var16', 'Var17', 'Var18', 'Var19', 'Var20', 'Var21','Var22', 'Var23', 'Var24', 'Var25', 'Var26', 'Var27', 'Var28', 'Var29',
                        'Var30', 'Var31', 'Var32', 'Var33', 'Var34', 'Var35', 'Var36', 'Var37', 'Var38', 'Var39', 'Var40', 'Var41', 'Var42', 'Var43',
                        'Var44', 'Var45', 'Var46', 'Var47', 'Var48', 'Var49', 'Var50', 'Var51', 'Var52', 'Var53', 'Var54', 'Var55']
    Temperature_variable = ['Temperature']
    Target_variables = ['Target']
    
    Numeric_variables.remove("Var55")       # 변수 삭제 -- 결측치 분석으로 삭제할 변수 선정함.
    
    df = df[Category_variables + Numeric_variables + Temperature_variable + Target_variables]
    df = df.dropna(axis = 0)                # 결측값 행 삭제
    
    # one-hot Encoding
    df[Category_variables] = df[Category_variables].astype("category")
    ohe_encoding = pd.get_dummies(df[Category_variables])
    df = df.drop(Category_variables, axis = 1)
    df = pd.concat([ohe_encoding, df], axis = 1)
    
    Category_variables = ohe_encoding.columns
    
    return df,Category_variables,Numeric_variables,Target_variables

def dataset_split(dataset):

    train_df = dataset.iloc[:int(len(dataset)*0.8),:].copy().reset_index(drop=True)
    vald_df = dataset.iloc[int(len(dataset)*0.8):int(len(dataset)*0.9),:].copy().reset_index(drop=True)
    test_df = dataset.iloc[int(len(dataset)*0.9):,:].copy().reset_index(drop=True)

    return train_df, vald_df, test_df

def show_Result(data_dict):
    
    df = pd.DataFrame(data_dict).T
    
    df.index.name = '구간'
    df.columns = ['MSE', 'RMSE', 'R² Score']
    
    print("###   전체 결과   ###")
    print(df)
    
    # 온도 1~5구간의 평균 계산
    mean_row = df.loc[
        ['온도 1구간(~1100)', '온도 2구간(1100~1120)', '온도 3구간(1120~1140)', '온도 4구간(1140~1160)', '온도 5구간(1160~)']
    ].mean()
    mean_row.name = '평균 (온도 1~5구간)'

    # 평균 행을 원래 df에 추가
    df = pd.concat([df, pd.DataFrame([mean_row])])

    # 보기 좋게 출력 (소수점 3자리로 반올림)
    print("###   평균 결과   ###")
    print(df.round(3))
    
    return