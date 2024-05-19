# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:22:51 2023

@author: ts
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")
# Đọc dử liệu từ tệp CSV và lưu vào DataFrame df
df = pd.read_csv("housing_price_dataset.csv") 
df.head()

#kiểm tra về kiểu dữ liệu
print(df.info())

# Kiểm tra các giá trị còn thiếu trong mỗi cột
missing_values = df.isnull().sum()
print(missing_values) 
# Kiểm tra giá trị trùng lặp
print("duplicated")
print(df.duplicated().sum())

print(df.Neighborhood.value_counts())
#Thống kê của df
print(df.describe().T)

#Biểu đồ Hộp
plt.figure(figsize=(10,10))
plt.boxplot(df.Price)
plt.title('Column: Price')


# Biểu đồ phân phối giá nhà theo khu phố
df = pd.DataFrame(df)
# Tính giá trung bình theo khu phố
average_price_by_neighborhood = df.groupby('Neighborhood')['Price'].mean()
# Vẽ biểu đồ tròn
plt.figure(figsize=(8, 8))
plt.pie(average_price_by_neighborhood, labels=average_price_by_neighborhood.index, autopct='%1.1f%%', startangle=140)
plt.title('Phân phối giá nhà theo khu phố')
plt.show()

#Biểu đồ heatmap
sns.heatmap(df.drop('Neighborhood', axis=1).corr(), annot=True)
dublicate_data = df[df.drop('Price', axis=1).duplicated(keep=False)].sort_values(['SquareFeet', 'Bedrooms', 'Neighborhood', 'YearBuilt', 'Price'])
print(f"Số lượng nhà có cùng tham số nhưng các chi phí khác nhau:{dublicate_data.shape[0]}")
print(dublicate_data.head(20))

# Xử lý dữ liệu
df.drop(df[df.Price <= 0].index, axis=0, inplace=True)
df.drop_duplicates(keep=False, inplace=True)
df.Neighborhood = LabelEncoder().fit_transform(df.Neighborhood)
print(df.head)

#mô hình #  Chia dữ liệu thành các tập (X_train, y_train) và kiểm tra (X_test, y_test).
X_train, X_test, y_train, y_test = train_test_split(df.drop("Price", axis=1), df.Price, 
    test_size=0.2, random_state=42)
 # Khởi tạo danh sách các mô hình
models = [
    MLPRegressor(),
    LinearRegression(),
    RandomForestRegressor()
]

best_model = None # Mô hình tốt nhất
best_score = None # Độ chính xác của mô hình tốt nhất
best_loss = None #  Độ sai lệch của mô hình tốt nhất
for clf in models: 
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)

    # Tính toán các chỉ số
    r2 = r2_score(y_test, y_pred) # Tính R2_score để đánh giá độ chính xác của mô hình.
    rmse = mean_squared_error(y_test, y_pred, squared=False) # Tính Root Mean Squared Error (RMSE) để đánh giá độ sai lệch của mô hình.
    print(f"{clf.__class__.__name__:30}: R2_score: {r2:17}, RMSE: {round(rmse, 6):10}")

    # Kiểm tra mô hình tốt nhất (chưa có tham số)
    if best_loss != None:
        if best_loss > rmse:
            best_model = clf
            best_score = r2
            best_loss = rmse
    else:
        best_model = clf
        best_score = r2
        best_loss = rmse

# Đưa ra mô hình tốt nhất (có tham số)
print("-"*92)
print(f"{best_model.__class__.__name__:30}: R2_score: {best_score}, RMSE: {round(best_loss, 6):10}")
print("\n")
print("\n")

#  chứa thông tin về các loại mô hình và các tham số cần tìm kiếm tốt nhất
model_params = {
    'MLPRegressor' : {
        'model': MLPRegressor(),
        'params': {
            "activation": ["identity", "logistic", "tanh", "relu"], 
            "solver": ["lbfgs", "sgd", "adam"], 
            "alpha": [0.00005,0.0005]
        }
    },
    'RandomForestRegressor': {
        'model': RandomForestRegressor(),
        'params' : {
            # Đây là số lượng cây quyết định sẽ được xây dựng trong mô hình RandomForestRegressor.
            'n_estimators': [60,100,300,500,700]
        }
    },
}
# Dùng  để duyệt qua từng mô hình và tập tham số tương ứng được định nghĩa trong
for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=3, return_train_score=False)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Tính toán các tham số
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"{model_name:30}: R2_score: {r2:17}, RMSE: {round(rmse, 6):10}")

    # Kiểm tra mô hình tốt nhất
    if best_loss != None:
        if best_loss > rmse:
            best_model = clf.best_estimator_
            best_score = r2
            best_loss = rmse
    else:
        best_model = clf.best_estimator_
        best_score = r2
        best_loss = rmse

# Chọn ra mô hình có tham số tốt nhất
print("-"*92)
print(f"{best_model.__class__.__name__:30}: R2_score: {best_score}, RMSE: {round(best_loss, 6):10}")









