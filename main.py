import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# Carregar os dados
data = pd.read_csv('data.csv')

# Pré-processar os dados
data['temperature'] = data['temperature'].astype(float)
data['wear'] = data['wear'].astype('category').cat.codes

# Dividir os dados em conjuntos de treino e teste
X = data[['temperature']]
y = data['wear']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cria um pipeline com escalonamento e classificador Random Forest
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Definir os hiperparâmetros para a busca em grade
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Usar StratifiedKFold para validação cruzada com n_splits=4
cv = StratifiedKFold(n_splits=4)

# Realizar a busca em grade com validação cruzada
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Melhor modelo encontrado
best_model = grid_search.best_estimator_

# Fazer previsões
y_pred = best_model.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Exibir os melhores hiperparâmetros
print(f'Best Hyperparameters: {grid_search.best_params_}')

# Salvar o modelo treinado
joblib.dump(best_model, 'wear_model.pkl')