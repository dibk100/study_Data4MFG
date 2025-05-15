import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def data_processing():
    dir_paht = "../data/PART_I/file.csv"
    
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
    """
    데이터셋 split : binary
    
    """
    X,Y_ = train_test_split(dataset, test_size=0.2, random_state=42)     # 학습 데이터와  테스트 데이터로 나눔.
    V,Y = train_test_split(Y_, test_size=0.5, random_state=42)
    return X.reset_index(drop=True),V.reset_index(drop=True),Y.reset_index(drop=True)

############ >> 온도별로 지표 확인하기 함수 만들기
