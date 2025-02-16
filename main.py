import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

# Treinar o modelo
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Fazer previsões
y_pred = model.predict(x_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

st.title("Prevendo o que deve ser vestido")
st.divider()

with st.form(key='my_form'):
    temperature = st.number_input("Digite a temperatura")
    submit_button = st.form_submit_button(label='Prever')

if submit_button:
    predicted_wear = label_encoder.inverse_transform(model.predict([[temperature]]))
    st.write(f"Para a temperatura de {temperature}°C, você deve vestir {predicted_wear[0]}")