import pandas as pd
df = pd.read_csv("FoodExpenditure.csv")
X = df.loc[:,["income"]]
Y = df.loc[:,"foodexp"]
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
model = linear_model.LinearRegression()
model.fit(X,Y)
model.coef_
model.intercept_
pred = model.predict(X)
rmse = mean_squared_error(Y,pred)**0.5
rmse
