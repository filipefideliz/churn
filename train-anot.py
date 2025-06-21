# %%
# METRICA SEM USAR O PIPLINE

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
