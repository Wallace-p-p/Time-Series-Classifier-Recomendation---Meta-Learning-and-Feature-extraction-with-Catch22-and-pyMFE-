from aeon.datasets import load_classification
from aeon.transformations.collection.feature_based import Catch22
from pymfe.mfe import MFE
from aeon.transformations.collection.feature_based import Catch22
import datetime
import pickle as pkl
import pandas as pd
import warnings 
warnings.simplefilter('ignore')


# Meta-Target -> Results from bake off redux for 34 TSC
from aeon.datasets.tsc_datasets import  univariate_equal_length
from aeon.benchmarking.published_results import load_classification_bake_off_2023_results
results, data, cls= load_classification_bake_off_2023_results(num_resamples=30, as_array=True)

# salvar meta-target
arr = [results, data, cls, univariate_equal_length]
with open('Bases/Meta-Target.pkl', 'wb') as f:
    pkl.dump(arr, f)
with open('Bases/Meta-Target.pkl', 'rb') as f:
    target = pkl.load(f)


# Meta-features com Catch22 + MFE
Tempos = []
meta_features = []
erros = []
for i in range(len(target[3])):
    try:
        t1 = datetime.datetime.now()
        print(i)
        # transformacao catch22 
        c22_no_nan = Catch22(replace_nans=True)
        dataset = target[3][i]
        X, y = load_classification(name=dataset)  
        data_no_nan = c22_no_nan.fit_transform(X)
        # transformacao do dataset transformado em meta features
        mfe = MFE(groups=["general", "statistical", "info-theory"])
        mfe.fit(data_no_nan,y)
        ft = mfe.extract()
        ft = [dataset, i, ft]
        # armazena as informacoes da meta feature em um list
        meta_features.append(ft)
        # registra o tempo consumido
        t2 = datetime.datetime.now()
        tempo = t2 - t1
        l = [dataset, str(tempo)]
        print(l)
        Tempos.append(l)

    except:
        erros.append(i)

# salva as meta features e o tempo em diferentes formatos
with open('Bases/Meta-Features_Catch22+MFE.pkl', 'wb') as f:
    pkl.dump(meta_features, f)
df = pd.DataFrame(meta_features)
df.to_excel('Bases/Meta-Features_Catch22+MFE.xlsx')
df.to_csv('Bases/Meta-Features_Catch22+MFE.csv')
df2 = pd.DataFrame(Tempos)
with open('Bases/Tempo_Catch22+MFE.pkl', 'wb') as f:
    pkl.dump(Tempos, f)
df2.to_excel('Bases/Tempo_Catch22+MFE.xlsx')
df2.to_csv('Bases/Tempo_Catch22+MFE.csv')









