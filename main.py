import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregar os dados
data = pd.read_csv('data.csv')

# Pré-processar os dados
data['temperature'] = data['temperature'].astype(float)
data['wear'] = data['wear'].astype('category').cat.codes

# Dividir os dados em conjuntos de treino e teste
X = data[['temperature']]
y = data['wear']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Exibir os coeficientes do modelo
print(f'Coeficientes: {model.coef_}')
print(f'Intercepto: {model.intercept_}')

# Salvar o modelo treinado
joblib.dump(model, 'wear_model.pkl')