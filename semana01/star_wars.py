# %%
import pandas as pd

df = pd.read_parquet("data/dados_clones.parquet")
df['General Jedi encarregado'].unique()


# %%
df.columns

# %%
features = [
    "Estatura(cm)",
    "Massa(em kilos)",
    "Distância Ombro a ombro",
    "Tamanho do crânio",
    "Tamanho dos pés",
]

cat_features = ["Distância Ombro a ombro",
                "Tamanho do crânio",
                "Tamanho dos pés"]   

target = 'Status '



X = df[features]
y = df[target]


# %%  e necessario instalar pip install feature-engine
from feature_engine import encoding
onehot = encoding.OneHotEncoder(variables=cat_features)
onehot.fit(X)
X = onehot.transform(X)
X

# %%
X

# %% max depth e a quantidade ramo da arvore se for 1 ela so vai fazer um ramo
from sklearn import tree
arvore = tree.DecisionTreeClassifier(max_depth=3)
arvore.fit(X, y)

# %%
# o filled true e para colocar a cor para diferenciar a pureza do no
#features pegando as variaveis do modelo como peso altura e etc...
#class name = para pegar o nome do alvo target segundo o que vo esta definindo como por exemplo nosso target e status entao definimos
# no class name pegamos o nome apto e defeituoso

import matplotlib.pyplot as plt
plt.figure(dpi=600)
tree.plot_tree(arvore,
               class_names=arvore.classes_,
               feature_names=features,
               filled=True,
               )