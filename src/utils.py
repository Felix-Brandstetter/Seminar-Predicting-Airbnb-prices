from sklearn.metrics import r2_score
from sklearn import metrics
import numpy as np

def print_scores(m, X_train, y_train, X_valid, y_valid):
    print('Train R2 = ', r2_score(y_train, m.predict(X_train, 10000)), 
          ', Valid R2 = ', r2_score(y_valid, m.predict(X_valid, 10000)), ', Valid MSE = ', 
          m.evaluate(X_valid, y_valid, 10000, 0))

def print_evaluate(Model, tt, Price, PredictedPrice):  
    from sklearn import metrics
    mae = metrics.mean_absolute_error(Price, PredictedPrice)
    mse = metrics.mean_squared_error(Price, PredictedPrice)
    rmse = np.sqrt(metrics.mean_squared_error(Price, PredictedPrice))
    r2_square = metrics.r2_score(Price, PredictedPrice)
    print(Model+' '+tt)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
