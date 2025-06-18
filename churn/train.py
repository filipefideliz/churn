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

sumario['diff_abs'] =  sumario[0] - sumario[1]
sumario['diff_rel'] = sumario[0] / sumario[1]
sumario.sort_values(by=['diff_rel'],ascending = False)

# %%
from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train,y_train)

# %% proporção
feature_importances =  (pd.Series(arvore.feature_importances_,index = X_train.columns).sort_values(ascending= False).reset_index())
feature_importances['acum.']= feature_importances[0].cumsum()
feature_importances[feature_importances['acum.'] < 0.96]
# %%

bets_features = ()


# %%
from feature_engine import discretisation

tree_discretization = discretisation.DecisionTreeDiscretiser(variables=best_features,   
                                                               regression=False,
                                                               cv=3,                 


)

tree_discretization.fit(X_train,y_train)
