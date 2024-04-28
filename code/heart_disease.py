"""
Heart disease detection (Must): The Heart Disease Prediction problem aims to develop a model that can accurately predict
 the likelihood of an individual having heart disease based on various medical and lifestyle factors. Given a dataset
 containing information such as age, sex, blood pressure, cholesterol levels, heart rate, and other relevant clinical
 indicators, the goal is to build a predictive model that can assist healthcare professionals in early detection and
 intervention for heart disease.
 Dataset Link: https://drive.google.com/file/d/1CEql-OEexf9p02M5vCC1RDLXibHYE9Xz/view
"""
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


filepath = r"..\data\heart_disease_data.csv"
heart_data = pd.read_csv(filepath)

heart_data_shuffled = heart_data.sample(frac=1, random_state=0)

# target
y = heart_data_shuffled.target

# parameters
params = [c for c in heart_data_shuffled.columns if c != "target"]
X = heart_data_shuffled[params]

scaler_min_max = MinMaxScaler()
X = scaler_min_max.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# splitting the data into training & testing datas
train_X, val_X, train_y, val_y = train_test_split(X, y,
                                                  train_size=0.85, test_size=0.15,
                                                  random_state=0)

model = LogisticRegression(random_state=0)
model.fit(train_X, train_y)
pred = model.predict(val_X)


print("----------F1 SCORE----------")
print(round(f1_score(val_y, pred), 2))

print("----------ACCURACY----------")
print(round(accuracy_score(val_y, pred), 2))

