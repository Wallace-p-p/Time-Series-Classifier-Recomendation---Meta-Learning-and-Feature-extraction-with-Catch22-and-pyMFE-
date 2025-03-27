import numpy as np
import pandas as pd
import pickle as pkl
import datetime
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import  LinearRegression
from aeon.datasets import load_classification
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import LeaveOneOut
import warnings 
warnings.simplefilter('ignore')

# Métricas

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

with open('Bases/Meta-Target.pkl', 'rb') as f:
    target = pkl.load(f)
with open('Bases/Meta-Features_Catch22+MFE.pkl', 'rb') as f:
    Metabase = pkl.load(f)
# Meta-knowledge
X = []
Y= []
for i in range(len(Metabase)):
    X.append(Metabase[i][2][1])
    Y.append(target[0][i])
X = np.asarray(X)
Y= np.asarray(Y)
Y = np.nan_to_num(Y, nan=0)
X = np.nan_to_num(X, nan=0)

# Limitação de valores muito grandes
X[X>100000] = 100000

# Listas, informacoes a serem usadas e registradas
Classifiers = target[2]
Regressors = ['SVR', 'KNN', 'Tree', 'RandomForest', 'MLP','linearBayesianRidge','linear']
Metrics= ['MAE','MAPE','MSquareE']

regs = [ svm.SVR(), KNeighborsRegressor(n_neighbors=3), tree.DecisionTreeRegressor(), RandomForestRegressor(random_state=0), MLPRegressor(random_state=1), linear_model.BayesianRidge(), LinearRegression()]

# Leave-one-out 
loo = LeaveOneOut()
loo.get_n_splits(X)

# Calculos das predicoes em leaveoneout 
TabelaFinal = []
Tempos = []
for rr in range(len(regs)): #  loop por regressor
        print(rr)
        L0 = [Regressors[rr]]
        t1 = datetime.datetime.now() # variavel inicial do tempo consumido
        try:
            for i, (train_index, test_index) in enumerate(loo.split(X)): # loop para cada split da leave one out
                xx = X[train_index]
                yy = Y[train_index]
                regr0 = regs[rr]
                regr = MultiOutputRegressor(regr0).fit(xx, yy) # treino do regrossor
                for jj in test_index: # loop para as predicoes do dataset separado na validacao
                    L1 = [target[3][jj]]
                    y_pred = regr.predict(X[jj].reshape(1,-1)) # predicao
                    y_true = Y[jj].reshape(1,-1)
                    # Clipping, limitacao de >1 para 1 e <0 para 0
                    lim = []
                    for zz in range(len(y_pred[0])):
                        if y_pred[0][zz] > 1:
                            lim.append(['1'])
                            y_pred[0][zz] = 1
                        elif y_pred[0][zz] < 0:
                            lim.append(['0'])
                            y_pred[0][zz] = 0
                        else:
                            lim.append(['no'])
                    # Calculo das metricas sobre a predicao de cada candidato
                    result1 = [mean_absolute_error(y_true, y_pred, multioutput='raw_values'),mean_absolute_percentage_error(y_true, y_pred,multioutput='raw_values'),mean_squared_error(y_true, y_pred, multioutput='raw_values')]
                    # Criacao da tabela de resultados, visando o a facilidade em ETL para a leitura em PowerBI pela power query
                    for kk in range(len(Metrics)):
                            L2 = [Metrics[kk]]
                            for ii in range(len(result1[0])):
                                L3 = [target[2][ii]]
                                L4 = lim[ii]
                                Lfinal = L0+L1+L2+L3+L4
                                Lfinal.append(result1[kk][ii])
                                TabelaFinal.append(Lfinal)
        except Exception as e: 
            print(i,': ->', e)
            print('Erro na iteração ', rr, '->', Regressors[rr])
        # Registro da variacao de tempo para cada regressor
        t2 = datetime.datetime.now()
        t= t2-t1
        Lt= [Regressors[rr], str(t)]
        Tempos.append(Lt)

# Salvamento dos resultados de tempo e das metricas na pasta Resultados
df = pd.DataFrame(TabelaFinal)
df.to_excel('Resultados/Tabela loo Limitada.xlsx')
df2 = pd.DataFrame(Tempos)
df2.to_excel('Resultados/Tempos loo Limitada.xlsx')


