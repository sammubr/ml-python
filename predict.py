import pandas as pd
import joblib
import argparse

# Configurar o parser de argumentos
parser = argparse.ArgumentParser(description='Predict wear based on temperature.')
parser.add_argument('temperatures', type=float, nargs='+', help='Temperatures to predict wear for')
args = parser.parse_args()

# Obter as temperaturas a partir dos argumentos
temperatures_to_predict = args.temperatures

# Carregar o modelo treinado
model = joblib.load('wear_model.pkl')

# Carregar os dados originais para obter o mapeamento de categorias
data = pd.read_csv('data.csv')
wear_categories = data['wear'].astype('category').cat.categories

# Fazer predições para cada temperatura
for temperature in temperatures_to_predict:
    # Pré-processar a temperatura
    X_new = pd.DataFrame({'temperature': [temperature]})

    # Fazer a predição
    prediction = model.predict(X_new)

    # Garantir que a predição esteja dentro do intervalo válido
    predicted_index = int(round(prediction[0]))
    predicted_index = max(0, min(predicted_index, len(wear_categories) - 1))

    # Mapear a predição para a categoria original
    predicted_wear = wear_categories[predicted_index]

    # Exibir o resultado
    print(f"Predicted wear for temperature {temperature}: {predicted_wear}")