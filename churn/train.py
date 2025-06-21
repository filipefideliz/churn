# %%
import pandas as pd
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import ensemble
from feature_engine import discretisation, encoding
from sklearn import pipeline
import mlflow
from sklearn import metrics
import matplotlib.pyplot as plt


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


## Discretizar
tree_discretization = discretisation.DecisionTreeDiscretiser(
    variables=best_features,
    regression=False,
    bin_output='bin_number',
    # cv=3 significa que o modelo será treinado e avaliado em 3 diferentes subconjuntos de dados.
    cv=3,
)

# Onehot
onehot = encoding.OneHotEncoder(variables=best_features, ignore_format=True)


# %%
# MODEL - MODIFY

# model = linear_model.LogisticRegression(penalty=None, random_state=42, max_iter=1000000)
# model = naive_bayes.BernoulliNB()
# model = ensemble.RandomForestClassifier(random_state=42,
#                                         min_samples_leaf=20,
#                                         n_jobs=-1,
#                                         n_estimators=500,
#                                         )

# model = tree.DecisionTreeClassifier(random_state=42, min_samples_leaf=20)
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment(experiment_id='541598652138216531')

with mlflow.start_run():

    mlflow.sklearn.autolog()

    model = ensemble.RandomForestClassifier(
        random_state=42,
        n_jobs=2,
    )

    params = {
        "min_samples_leaf":[15,20,25,30,50],
        "n_estimators":[100,200,500,1000],
        "criterion":['gini', 'entropy', 'log_loss'],
    }

    grid = model_selection.GridSearchCV(model,
                                        params,
                                        cv=3,
                                        scoring='roc_auc',
                                        verbose=4,
                                        )

    model_pipeline = pipeline.Pipeline(
        steps=[
            ('Discretizar', tree_discretization),
            ('Onehot', onehot),
            ('Grid',grid), 
        ]
    )

    model_pipeline.fit(X_train[best_features], y_train)

    ## ASSESS
    # faz a prediçao das classes previstas para cada amostra de entrada
    y_train_predict = model_pipeline.predict(X_train[best_features])
    # Esta linha calcula as probabilidades de que cada amostra do conjunto de treino pertença a cada classe
    # [: ,1] seleciono todas as linha da matriz buscando o 1   
    y_train_proba = model_pipeline.predict_proba(X_train[best_features])[:,1]

    acc_train = metrics.accuracy_score(y_train, y_train_predict)
    auc_train = metrics.roc_auc_score(y_train, y_train_proba)
    roc_train = metrics.roc_curve(y_train, y_train_proba)
    print("Acurácia Treino:", acc_train)
    print("AUC Treino:", auc_train)

    y_test_predict = model_pipeline.predict(X_test[best_features])
    y_test_proba = model_pipeline.predict_proba(X_test[best_features])[:,1]

    acc_test = metrics.accuracy_score(y_test, y_test_predict)
    auc_test = metrics.roc_auc_score(y_test, y_test_proba)
    roc_test = metrics.roc_curve(y_test, y_test_proba)
    print("Acurácia Test:", acc_test)
    print("AUC Test:", auc_test)

    y_oot_predict = model_pipeline.predict(oot[best_features])
    y_oot_proba = model_pipeline.predict_proba(oot[best_features])[:,1]

    acc_oot = metrics.accuracy_score(oot[target], y_oot_predict)
    auc_oot = metrics.roc_auc_score(oot[target], y_oot_proba)
    roc_oot = metrics.roc_curve(oot[target], y_oot_proba)
    print("Acurácia oot:", acc_oot)
    print("AUC oot:", auc_oot)

    mlflow.log_metrics({
    "acc_train":acc_train,
    "auc_train":auc_train,
    "acc_test":acc_test,
    "auc_test":auc_test,
    "acc_oot":acc_oot,
    "auc_oot":auc_oot,
    })

# %%

plt.figure(dpi=400)
plt.plot(roc_train[0], roc_train[1])
plt.plot(roc_test[0], roc_test[1])
plt.plot(roc_oot[0], roc_oot[1])
plt.plot([0,1], [0,1], '--', color='black')
plt.grid(True)
plt.ylabel("Sensibilidade")
plt.xlabel("1 - Especificidade")
plt.title("Curva ROC")
plt.legend([
    f"Treino: {100*auc_train:.2f}",
    f"Teste: {100*auc_test:.2f}",
    f"Out-of-Time: {100*auc_oot:.2f}",
])

plt.show() 

