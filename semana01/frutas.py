# %%
import pandas as pd

df = pd.read_excel("data/dados_frutas.xlsx") 
df

# %% 
from sklearn import tree

arvore =tree.DecisionTreeClassifier(random_state=42)

# %%

y = df['Fruta']
caracteristica = ['Arredondada','Suculenta','Vermelha','Doce']

X = df[caracteristica]
# %%
arvore.fit(X,y)

# %%

arvore.predict([[1,1,1,1]])

# %%
import matplotlib.pyplot as plt

plt.figure(dpi= 400 , figsize=[4,4])

tree.plot_tree(
    arvore,
    feature_names=caracteristica,
    class_names=arvore.classes_,
    filled=True
)

# %%

proba = arvore.predict_proba([[1,1,1,1]])[0]
pd.Series(proba, index=arvore.classes_)