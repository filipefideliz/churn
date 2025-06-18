# %%

import pandas as pd

df = pd.read_csv("../data/abt_churn.csv")
df.head()

pd.options.display.max_columns = 500
pd.options.display.max_rows = 500

# %%
df['dtRef'].sort_values().unique()
# %%
df['dtRef'].value_counts().sort_index()
# %% 
oot= df[df['dtRef'] == df['dtRef'].max()].copy()
oot
# %%
df_train = df[df['dtRef'] < df['dtRef'].max()].copy()
df_train['dtRef']
# %%
features = df_train.columns[2:-1]
target = 'flagChurn'

X, y = df_train[features], df_train[target]

# %% semma => sample
# stratify = garantir uma taxa proxima do modelo que queremos tenham exatamente a taxa de resposta que pedimos
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split( X ,y , 
                                                                    random_state=42,
                                                                    test_size=0.2,
                                                                    stratify=y,
                                                                    )

# %%
print('taxa variavel teste: ',y_train.mean())
print('taxa variavel teste: ',y_test.mean())

# %% apartir daqui so usamos a base de treino
# %% EXPLORE (MISSINGS)

X_train.isna().sum().sort_values(ascending = False)

# %% ANALIES BI VARIADA -DESCUBRIR AS FEATURES QUE AJUDA AGENTE A PREVER A REPOSTA

df_analise = X_train.copy()
df_analise[target] = y_train
sumario = df_analise.groupby(by=target).agg(['mean','median']).T
sumario

# %%
## calculo a diferença entre a media e a mediana
sumario['diff_abs'] =  sumario[0] - sumario[1]
## calculo a proporçao entre elas
sumario['diff_rel'] = sumario[0] / sumario[1]

## para indentificar das variavies mais se diferenciam das outras
sumario.sort_values(by=['diff_rel'],ascending = False)

# %%
from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train,y_train)
# %% calculo a importancia das features da arvore de desicao
feature_importances =  (pd.Series(arvore.feature_importances_,index = X_train.columns).sort_values(ascending= False).reset_index())
feature_importances['acum.']= feature_importances[0].cumsum()
feature_importances[feature_importances['acum.'] < 0.96]
# %%
##['index'] Após filtrar, seleciona apenas a coluna com os nomes das features ('index'), que contém os nomes das variáveis.

best_features = (feature_importances[feature_importances['acum.'] < 0.96]['index']
                 .tolist())

best_features

# %%
# MODIFY

from feature_engine import discretisation

tree_discretization = discretisation.DecisionTreeDiscretiser(
    variables=best_features,
    regression=False,
    bin_output='bin_number',
    # cv=3 significa que o modelo será treinado e avaliado em 3 diferentes subconjuntos de dados.
    cv=3,
)

## ensinando o modelo a criar a discretização
tree_discretization.fit(X_train[best_features], y_train)

# %%

## converte os valores contínuos em seus respectivos números de bin.
X_train_transform = tree_discretization.transform(X_train[best_features])
X_train_transform

# %%
# MODEL
from sklearn import linear_model

reg = linear_model.LogisticRegression(penalty=None, random_state=42, max_iter=1000000)
reg.fit(X_train_transform, y_train)

# %%
from sklearn import metrics

# faz a prediçao das classes previstas para cada amostra de entrada
y_train_predict = reg.predict(X_train_transform)
# Esta linha calcula as probabilidades de que cada amostra do conjunto de treino pertença a cada classe
y_train_proba = reg.predict_proba(X_train_transform)[:,1]

acc_train = metrics.accuracy_score(y_train, y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_proba)
print("Acurácia Treino:", acc_train)
print("AUC Treino:", auc_train)


X_test_transform = tree_discretization.transform(X_test[best_features])

y_test_predict = reg.predict(X_test_transform)
y_test_proba = reg.predict_proba(X_test_transform)[:,1]

acc_test = metrics.accuracy_score(y_test, y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)
print("Acurácia Test:", acc_test)
print("AUC Test:", auc_test)

oot_transform = tree_discretization.transform(oot[best_features])

y_oot_predict = reg.predict(oot_transform)
# [: ,1] seleciono todas as linha da matriz buscando o 1 
y_oot_proba = reg.predict_proba(oot_transform)[:,1]

acc_oot = metrics.accuracy_score(oot[target], y_oot_predict)
auc_oot = metrics.roc_auc_score(oot[target], y_oot_proba)
print("Acurácia oot:", acc_oot)
print("AUC oot:", auc_oot)
# %%