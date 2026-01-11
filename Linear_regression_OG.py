import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


data = {
    'Hours_Studied': [1, 2, 3, 4, 5],
    'Marks': [40, 50, 60, 70, 80]
}

df = pd.DataFrame(data)

X = df[['Hours_Studied']]
y = df['Marks']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)

#predicting
y_pred = model.predict(X_test)

#evaling the results
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
