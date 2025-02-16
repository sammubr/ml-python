import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar os dados
data = pd.read_csv('data.csv')

# Pré-processar os dados
data['temperature'] = data['temperature'].astype(float)
label_encoder = LabelEncoder()
data['wear'] = label_encoder.fit_transform(data['wear'])

# Dividir os dados em conjuntos de treino e teste
x = data[['temperature']]
y = data['wear']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Definir a grade de parâmetros
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Treinar o modelo
rf = RandomForestClassifier()

# Inicializar o RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Ajustar o modelo
rf_random.fit(x_train, y_train)

# Melhor combinação de parâmetros
best_params = rf_random.best_params_
print(f'Melhores parâmetros: {best_params}')

# Treinar o modelo com os melhores parâmetros
best_model = rf_random.best_estimator_
best_model.fit(x_train, y_train)

# Fazer previsões
y_pred = best_model.predict(x_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Definir os nomes das classes
target_names = label_encoder.classes_

# Gerar a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Exibir o gráfico no Streamlit
st.pyplot(plt)

st.title("Prevendo o que deve ser vestido")
st.divider()

with st.form(key='my_form'):
    temperature = st.number_input("Digite a temperatura")
    submit_button = st.form_submit_button(label='Prever')

if submit_button:
    predicted_wear = label_encoder.inverse_transform(best_model.predict([[temperature]]))
    st.write(f"Para a temperatura de {temperature}°C, você deve vestir {predicted_wear[0]}")