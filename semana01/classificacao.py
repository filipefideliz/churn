# %%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df

df['aprovado'] = (df['nota'] > 5).astype(int)
df

# %%

plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
plt.grid(True)
plt.title("Cervaja vs Aprovação")
plt.xlabel("Cervejas")
plt.ylabel("Aprovado")

# %%

from sklearn import linear_model
from sklearn import tree
from sklearn import naive_bayes

reg = linear_model.LogisticRegression(penalty=None,
                                      fit_intercept=True)
reg.fit(df[['cerveja']], df['aprovado'])
reg_predict = reg.predict(df[['cerveja']].drop_duplicates())
reg_prob = reg.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

arvore_full = tree.DecisionTreeClassifier(random_state=42)
arvore_full.fit(df[['cerveja']], df['aprovado'])
arvore_full_predict = arvore_full.predict(df[['cerveja']].drop_duplicates())
arvore_full_proba = arvore_full.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

arvore_d2 = tree.DecisionTreeClassifier(random_state=42, max_depth=2)
arvore_d2.fit(df[['cerveja']], df['aprovado'])
arvore_d2_predict = arvore_d2.predict(df[['cerveja']].drop_duplicates())
arvore_d2_proba = arvore_d2.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

nb = naive_bayes.GaussianNB()
nb.fit(df[['cerveja']], df['aprovado'])
nb_predict = nb.predict(df[['cerveja']].drop_duplicates())
nb_proba = nb.predict_proba(df[['cerveja']].drop_duplicates())[:,1]


plt.figure(dpi=400)
plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
plt.grid(True)
plt.title("Cervaja vs Aprovação")
plt.xlabel("Cervejas")
plt.ylabel("Aprovado")
plt.plot(df['cerveja'].drop_duplicates(), reg_predict, color='tomato')
plt.plot(df['cerveja'].drop_duplicates(), reg_prob, color='red')

plt.plot(df['cerveja'].drop_duplicates(), nb_predict, color='green')
plt.plot(df['cerveja'].drop_duplicates(), nb_proba, color='magenta')

plt.plot(df['cerveja'].drop_duplicates(), arvore_d2_predict, color='blue')
plt.plot(df['cerveja'].drop_duplicates(), arvore_d2_proba, color='black')

plt.hlines(0.5, xmin=1, xmax=9, linestyles='--', colors='black')

plt.legend(['Oberservação',
            "Reg Predict",
            "Reg Proba",
            "Naive Bayes Predict",
            "Naive Bayes Proba",
            "Árvore D2 Predict",
            "Árvore D2 Proba",
            ])
# %%

