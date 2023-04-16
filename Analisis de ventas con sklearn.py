import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split




data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data.drop('Sales', axis=1), data['Sales'], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = model.score(X_test, y_test)

print(f'Predicciones: {y_pred}')
print(f'Rendimiento del modelo: {score}')





