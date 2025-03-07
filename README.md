# Pipeline-com-Scikit-Learn-para-Prever-o-Pre-o-de-Autom-veis
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Dados fictícios
data = {
    'Combustível': ['Gasolina', 'Diesel', 'Etanol', 'Gasolina', 'Diesel'],
    'Idade': [5, 3, 7, 2, 4],
    'Quilometragem': [50000, 40000, 60000, 30000, 45000],
    'Preço': [30000, 35000, 25000, 40000, 32000]
}

df = pd.DataFrame(data)

# Separando variáveis independentes (X) e dependente (y)
X = df.drop('Preço', axis=1)
y = df['Preço']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo as transformações
categorical_features = ['Combustível']
numerical_features = ['Idade', 'Quilometragem']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ])

# Criando o pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Treinando o modelo
pipeline.fit(X_train, y_train)

# Fazendo previsões
y_pred = pipeline.predict(X_test)

# Avaliando o modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Erro Quadrático Médio (MSE): {mse}")
