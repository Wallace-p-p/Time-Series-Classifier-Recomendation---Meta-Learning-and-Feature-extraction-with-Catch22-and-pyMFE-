import pickle as pkl
import numpy as np
import pandas as pd
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

# Meta-target Abanda
classifier_list = np.genfromtxt("Comparação/classifier_list.txt",dtype="str")
results_UCR = np.loadtxt("Comparação/results_UCR.txt")
UCR_list = np.genfromtxt('Comparação/db_names.txt',dtype='str')

# Meta-feature landmarkers Abanda
landmarkers_UCR = np.loadtxt("Comparação/landmarkers_UCR.txt")
X1 = landmarkers_UCR
Y= np.asarray(results_UCR)
print(len(X1),'x',len(X1[0]),' ---- target:', len(Y),'x',len(Y[0]))
Y = np.nan_to_num(Y, nan=0)
X1 = np.nan_to_num(X1, nan=0)

# meta-feature mfe Abanda
metafeatures = np.zeros((len(UCR_list),73))
for i in range(len(UCR_list)):
    metafeatures[i,:]=np.loadtxt("Comparação/mf/mf_"+UCR_list[i]+'.txt')
X0 = metafeatures
print(len(X0),'x',len(X0[0]),' ---- target:', len(Y),'x',len(Y[0]))
X0 = np.nan_to_num(X0, nan=0)

# Meta-features tcc
with open('Bases/Meta-Features_Catch22+MFE.pkl', 'rb') as f:
    Metabase = pkl.load(f)
with open('Bases/Meta-Target.pkl', 'rb') as f:
    target = pkl.load(f)


# Reorganização das linhas
X2 = []
for i in range(len(Metabase)):
    X2.append(Metabase[i][2][1])
X2 = np.asarray(X2)
print(len(X2),'x',len(X2[0]))
X2 = np.nan_to_num(X2, nan=0)
Xreordenado = []
for i in range(len(UCR_list)):
    Xreordenado.append(X2[target[3].index(UCR_list[i])])
X2 = np.asarray(Xreordenado)
print(len(X2),'x',len(X2[0]),' ---- target:', len(Y),'x',len(Y[0]))
X2 = np.nan_to_num(X2, nan=0)

# Listas
Classifiers = classifier_list
Regressors = ['SVR', 'KNN', 'Tree', 'RandomForest', 'MLP', 'linearBayesianRidge', 'Linear']
Metrics= ['MAE','MAPE','MSquareE']
Bases = [ 'mfe', 'LandMarkers', 'Catch22+mfe']

# Padronização de valores muito grandes
X0[X0>100000] = 100000
X1[X1>100000] = 100000
X2[X2>100000] = 100000
Xt=[X0,X1,X2]

regs = [ svm.SVR(), KNeighborsRegressor(n_neighbors=3), tree.DecisionTreeRegressor(), RandomForestRegressor(random_state=0), MLPRegressor(random_state=1), linear_model.BayesianRidge(), LinearRegression()]

Tempos =[]
for db in range(len(Xt)):
    X=Xt[db]
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    TabelaFinal = []
    for rr in range(len(regs)):
            t1 = datetime.datetime.now()
            L0 = [Regressors[rr]]
            try:
                for i, (train_index, test_index) in enumerate(loo.split(X)):
                    xx = X[train_index]
                    yy = Y[train_index]
                    regr0 = regs[rr]
                    regr = MultiOutputRegressor(regr0).fit(xx, yy)
                    for jj in test_index:
                        L1 = [target[3][jj]]
                        y_pred = regr.predict(X[jj].reshape(1,-1))
                        y_true = Y[jj].reshape(1,-1)
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
                        result1 = [mean_absolute_error(y_true, y_pred, multioutput='raw_values'),mean_absolute_percentage_error(y_true, y_pred,multioutput='raw_values'),mean_squared_error(y_true, y_pred, multioutput='raw_values')]
                        for kk in range(len(Metrics)):
                                L2 = [Metrics[kk]]
                                for ii in range(len(result1[0])):
                                    L3 = [Classifiers[ii]]
                                    L4 = [lim[ii]]
                                    Lfinal = L0+L1+L2+L3+L4
                                    Lfinal.append(result1[kk][ii])
                                    TabelaFinal.append(Lfinal)       
            except Exception as e: 
                print(i,': ->', e)
                print('Erro na iteração ', rr, '->', Regressors[rr])
            t2 = datetime.datetime.now()
            t = t2-t1
            Lt = [ Bases[db], Regressors[rr], str(t)]
            Tempos.append(Lt) 
    df = pd.DataFrame(TabelaFinal)
    df.to_excel('Resultados Comparacao/Tabela loo Comparação '+str(db)+'.xlsx')
df2 = pd.DataFrame(Tempos)
df2.to_excel('Resultados Comparacao/Tempos '+Bases[db]+'.xlsx')

