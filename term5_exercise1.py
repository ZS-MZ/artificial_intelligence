import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score





# ==============================
# <<<<<<<<<< بخش اول >>>>>>>>>>
# ==============================





# 1) و بررسی کلی آن csv وارد کردن داده های فایل :

# csv وارد کردن داده های فایل
data = pd.read_csv('loans.csv')

# بررسی اولیه داده ها
# print(data.head(), end='\n================================================================================\n')
# print(data.info(), end='\n================================================================================\n')
# print(data.describe(), end='\n================================================================================\n')
# _________________________________________________________________________________________________________________________________________





# 2) تشخیص و پردازش مقادیر گم شده :

# تشخیص مقادیر گم شده
# missing_values = data.isnull().sum()
# print(missing_values, end='\n================================================================================\n')

# نمایش درصد مقادیر گم شده در هر ستون
# print((missing_values / data.shape[0])*100, end='\n================================================================================\n')

# جایگزینی مقادیر گم شده در ستون مقدار وام با میانگین
mean_value = data['loan_amount'].mean()
data['loan_amount'].fillna(mean_value, inplace=True)

# حذف سطر هایی که مقادیر گم شده دارند
data.dropna(inplace=True)
# _________________________________________________________________________________________________________________________________________





# 3) تشخیص و پردازش مقادیر پرت :

# IQR شناسایی مقادیر پرت با استفاده از 
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound)   |   (data[column] > upper_bound)]
    return outliers

# شناسایی مقادیر پرت در ستون مقدار وام
# print(detect_outliers_iqr(data, 'loan_amount'), end='\n================================================================================\n')

# حذف مقادیر پرت در ستون مقدار وام
Q1 = data['loan_amount'].quantile(0.25)
Q3 = data['loan_amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['loan_amount'] >= lower_bound)   &   (data['loan_amount'] <= upper_bound)]
# _________________________________________________________________________________________________________________________________________





# 4) تبدیل متغیر ها :

# تبدیل لگاریتمی متغیر مقدار وام
data['log_loan_amount'] = np.log1p(data['loan_amount'])

# تبدیل رادیکالی متغیر مقدار وام
data['sqrt_loan_amount'] = np.sqrt(data['loan_amount'])
# _________________________________________________________________________________________________________________________________________





# 5) مقیاس بندی متغیر های عددی :

# استاندارد سازی متغیر های عددی
# scaler = StandardScaler()
# data[['loan_amount','rate']] = scaler.fit_transform(data[['loan_amount','rate']])

# نمایش داده های استاندارد شده
# print(data[['loan_amount','rate']].head(), end='\n================================================================================\n')

# نرمال سازی متغیر های عددی
scaler = MinMaxScaler()
data[['loan_amount','rate']] = scaler.fit_transform(data[['loan_amount','rate']])

# نمایش داده های نرمال شده
# print(data[['loan_amount','rate']].head(), end='\n================================================================================\n')
# _________________________________________________________________________________________________________________________________________





# 6) رمزگذاری متغیر های دسته ای :

# Label Encoding
# label_encoder = LabelEncoder()
# data[['loan_type']] = label_encoder.fit_transform(data[['loan_type']])
# print(data[['loan_type']].head(), end='\n================================================================================\n')

# OneHot Encoding
data = pd.get_dummies(data, columns=['loan_type'])
# print(data.head(), end='\n================================================================================\n')
# _________________________________________________________________________________________________________________________________________





# 7) ایجاد متغیر های جدید :

# 7_1) ایجاد ویژگی تعاملی مقدار وام * نرخ سود
data['loan_rate'] = data['loan_amount'] * data['rate']

# 7_2) ایجاد ویژگی های چند جمله ای از درجه2
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(data[['loan_amount','rate']])

# اضافه کردن ویژگی های چند جمله ای جدید به دیتاست
poly_features_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['loan_amount','rate']))
data = pd.concat([data, poly_features_df], axis=1)
# print(data.head(), end='\n================================================================================\n')

# 7_3) datetime ایجاد ویژگی های زمانی با  تبدیل ستون های تاریخی به فرمت
data['loan_start'] = pd.to_datetime(data['loan_start'])
data['loan_end'] = pd.to_datetime(data['loan_end'])

# ایجاد ویژگی مدت زمان وام برحسب روز
data['loan_time'] = (data['loan_end'] - data['loan_start']).dt.days
# print(data[['loan_start', 'loan_end', 'loan_time']].head(), end='\n================================================================================\n')
# _________________________________________________________________________________________________________________________________________





# 8) تقسیم داده به مجموعه آموزش و آزمون :

# با هدف بودن ستون پرداخت شده (y) و برچسب ها (x) تعریف ویژگی ها
x = data.drop('repaid', axis=1)
y = data['repaid']

# تقسیم داده ها به مجموعه های آموزش و آزمون
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# نمایش اندازه های مجموعه های آموزش و آزمون
# print(f"Training set size: {x_train.shape[0]} samples") 
# print(f"Test set size: {x_test.shape[0]} samples", end='\n================================================================================\n') 
# _________________________________________________________________________________________________________________________________________





# ==============================
# <<<<<<<<<< بخش دوم >>>>>>>>>>
# ==============================





# 1) رگرسیون خطی یک متغیره :

# تعریف ویژگی و برچسب
x = data[['rate']]
y = data['loan_amount']

# تقسیم داده‌ها به مجموعه‌های آموزش و آزمون
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ایجاد مدل رگرسیون خطی
model = LinearRegression()
model.fit(x_train, y_train)

# پیش‌ بینی مقادیر
y_pred = model.predict(x_test)

# مقایسه مقادیر پیش‌ بینی شده و واقعی
# plt.scatter(x_test, y_test, color='blue', label='مقادیر واقعی')
# plt.plot(x_test, y_pred, color='red', linewidth=2, label='مقادیر پیش بینی شده')
# plt.xlabel('rate')
# plt.ylabel('loan_amount')
# plt.legend()
# plt.show()

# محاسبه خطای مدل
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# print(f"R-squared: {r2}", end='\n================================================================================\n')
# _________________________________________________________________________________________________________________________________________





# 2) رگرسیون خطی چند متغیره :

# آماده‌سازی داده‌ها
# x = data['rate', 'loan_time', 'loan_type']
# y = data['loan_amount']

# تقسیم داده‌ها به مجموعه‌های آموزش و آزمون
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ایجاد مدل رگرسیون خطی
# model = LinearRegression()
# model.fit(x_train, y_train)

# پیش‌ بینی مقادیر
# y_pred = model.predict(x_test)

# محاسبه خطا و ارزیابی مدل
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")
# print(f"R-squared: {r2}", end='\n================================================================================\n')

# نمایش مقادیر پیش‌بینی شده و واقعی
# comparison = pd.DataFrame({'واقعی': y_test, 'پیش‌ بینی شده': y_pred})
# print(comparison.head(), end='\n================================================================================\n')
# _________________________________________________________________________________________________________________________________________





# 3) رگرسیون چند جمله ای :

# 3_1) رگرسیون چند جمله ای درجه2

# انتخاب ویژگی‌ها
# x = data['rate', 'loan_time']
# y = data['loan_amount']

# ایجاد ویژگی‌های چند جمله‌ای درجه2
# poly2 = PolynomialFeatures(degree=2)
# x_poly2 = poly2.fit_transform(x)

# ایجاد مدل رگرسیون چند جمله‌ای درجه2
# model2 = LinearRegression()
# model2.fit(x_poly2, y)

# پیش‌ بینی مقادیر
# y_pred2 = model2.predict(x_poly2)

# مقایسه مقادیر پیش‌ بینی شده با مقادیر واقعی
# plt.scatter(y, y_pred2, color='blue')
# plt.xlabel('واقعی')
# plt.ylabel('پیش‌ بینی شده')
# plt.title('رگرسیون چند جمله‌ای درجه2')
# plt.show()

# ========

# 3_2) رگرسیون چند جمله ای درجه3

# انتخاب ویژگی‌ها
# x = data['rate', 'loan_time', 'loan_type']
# y = data['loan_amount']

# ایجاد ویژگی‌های چند جمله‌ای درجه3
# poly3 = PolynomialFeatures(degree=3)
# x_poly3 = poly3.fit_transform(x)

# ایجاد مدل رگرسیون چند جمله‌ای درجه3
# model3 = LinearRegression()
# model3.fit(x_poly3, y)

# پیش‌ بینی مقادیر
# y_pred3 = model3.predict(x_poly3)

# مقایسه مقادیر پیش‌ بینی شده با مقادیر واقعی
# plt.scatter(y, y_pred3, color='blue')
# plt.xlabel('واقعی')
# plt.ylabel('پیش‌بینی شده')
# plt.title('رگرسیون چند جمله‌ای درجه3')
# plt.show()
# _________________________________________________________________________________________________________________________________________





# 4) ارزیابی مدل :

# 4_1) ارزیابی مدل رگرسیون خطی یک متغیره
# mse_single = mean_squared_error(y_test, 'rate')
# r2_single = r2_score(y_test, 'rate')
# print(f'MSE یک متغیره: {mse_single}')
# print(f'R² یک متغیره: {r2_single}', end='\n================================================================================\n')

# 4_2) ارزیابی مدل رگرسیون خطی چند متغیره
# mse_multiple = mean_squared_error(y_test, 'rate', 'loan_time')
# r2_multiple = r2_score(y_test, 'rate', 'loan_time')
# print(f'MSE چند متغیره: {mse_multiple}')
# print(f'R² چند متغیره: {r2_multiple}', end='\n================================================================================\n')

# 4_3) ارزیابی مدل رگرسیون چند جمله‌ای
# mse_poly = mean_squared_error(y_test, data['rate', 'loan_time', 'loan_type'])
# r2_poly = r2_score(y_test, data['rate', 'loan_time', 'loan_type'])
# print(f'MSE چند جمله‌ای: {mse_poly}')
# print(f'R² چند جمله‌ای: {r2_poly}', end='\n================================================================================\n')
