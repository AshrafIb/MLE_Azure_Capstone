import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import MaxAbsScaler

import argparse 
import os 

from azureml.core.run import Run
from azureml.core import Workspace, Experiment, Datastore, Dataset 
from azureml.data.dataset_factory import TabularDatasetFactory


def preprocessing(data):
    '''
    Preprocesses the data by applying dummy-encoding,
    performing a train test split and standardizing.
    
    Input: 
        Data-File containing the Abalone Dataset 
    Output:
        x_train : training data 
        x_test  : testing data 
        y_train : training label 
        y_test  : testing label 
    '''
    df=data.to_pandas_dataframe()
    df.columns=['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
    df_dum=pd.get_dummies(df, columns=['Sex'],dummy_na=False)
    
    y_list=['Rings']
    x_list = [x for x in df_dum.columns if x not in y_list]

    x = df_dum[x_list]
    y = df_dum[y_list]

    x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.33, random_state=42, shuffle=True)

    transformer = MaxAbsScaler().fit(x_train)
    x_train=transformer.transform(x_train)
    x_test=transformer.transform(x_test)

    return x_train,x_test,y_train,y_test

def main():
    run = Run.get_context()

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help='Number of Estimators of Regressor')
    parser.add_argument('--max_depth', type=int, default=None, help='Depth of Tree and Tree Pruning')
    parser.add_argument('--max_features',type=str, default='auto', help='The Number of feautrues to use')
    parser.add_argument('--oob_score',type=bool, default=False, help='Out of Bag-Score, to compensate Bootstrap Missings')

    args = parser.parse_args()

    run.log("Estimators:",args.n_estimators)
    run.log("Depth:",args.max_depth)
    run.log('Features:',args.max_features)
    run.log('OoB:',args.oob_score)

    path='https://raw.githubusercontent.com/AshrafIb/MLE_Azure_Capstone/main/abalone.data'

    data=TabularDatasetFactory.from_delimited_files(path)
    
    x_train,x_test,y_train,y_test=preprocessing(data)

    rf_reg=RandomForestRegressor(n_estimators=args.n_estimators,max_depth=args.max_depth,
                                max_features=args.max_features,bootstrap=True, oob_score=args.oob_score,
                                n_jobs=-1).fit(x_train,y_train)

    r2_score=rf_reg.score(x_test,y_test)
    run.log('R2-Score',np.float(r2_score))

    # Calculating MAE and parsing to logs
    y_pred=rf_reg.predict(x_test)
    mae=mean_absolute_error(y_test,y_pred)
    run.log('MAE',np.float(mae))

if __name__=='__main__':
    main()
    

                        



