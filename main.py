import pandas as pd
import math
df = pd.DataFrame({
    'Месяц\год': ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь'],
    '2017': [65.000, 61.000, 63.000, 69.000, 70.580, 97.365, 104.755, 101.820, 83.655, 77.910, 70.365, 64.200],
    '2018': [69.550, 65.270, 67.410, 73.830, 75.521, 104.181, 112.088, 108.947, 89.511, 83.364, 75.291, 68.694],
    '2019': [71.358, 66.967, 69.163, 75.750, 77.484, 106.889, 115.002, 111.780, 91.838, 85.531, 77.248, 70.480],
    '2020': [77.781, 72.994, 75.387, 82.567, 84.458, 116.509, 125.352, 121.840, 100.104, 93.229, 84.200, 76.823],
    '2021': [81.670, 76.644, 79.157, 86.695, 88.681, 122.335, 131.620, 127.932, 105.109, 97.890, 88.410, 80.664],
    '2022': [89.837, 84.308, 87.072, 95.365, 97.549, 134.568, 144.782, 140.725, 115.620, 107.679, 97.252, 88.731],
    '2023': [97.826, 91.806, 94.816, 103.846, 106.224, 146.536, 157.658, 153.241, 125.902, 117.256, 105.900, 96.622],
    '2024': [106.804, 100.231, 103.518, 113.377, 115.973, 159.984, 172.127, 167.304, 137.457, 128.017, 115.619, 105.489]
})
pd.set_option('display.max_columns', None)
print(df)



print('Оптимистичный прогноз на 2023 и 2024 год:')

d = {'2023': pd.Series([111.298, 105.278, 108.288, 117.318, 119.696, 160.008, 171.130, 166.713, 139.374, 130.728, 119.373, 110.094],
                            index = ['январь', 'фeвраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь']),
           '2024': pd.Series([121.513, 114.940, 118.227, 128.085, 130.682, 174.693, 186.836, 182.013, 152.166, 142.726, 130.328, 120.198],
                            index = ['январь', 'фeвраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь'])}
df = pd.DataFrame(d)
print(df)



print('Пессимистичный прогноз на 2023 и 2024 год:')

d = {'2023': pd.Series([84.354, 78.334, 81.344, 90.374, 92.752, 133.063, 144.185, 139.768, 112.430, 103.783, 92.428, 83.150],
                            index = ['январь', 'фeвраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь']),
           '2024': pd.Series([92.095, 85.523, 88.809, 98.668, 101.264, 145.275, 157.418, 152.596, 122.748, 113.308, 100.911, 90.781],
                            index = ['январь', 'фeвраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь'])}
df = pd.DataFrame(d)
print(df)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print('---2023 YEAR---')

x = np.array([2017,2018,2019,2020,2021,2022]).reshape((-1, 1))
y = np.array([63.400, 67.838, 69.602,75.866,79.659,87.625])
x_log = np.log(x)
y_log = np.log(y)
model = LinearRegression()
model.fit(x_log, y_log)
rez = model.score(x_log,y_log)
print('Коэффициент детерминации:', rez)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
print('логарифмируемый прогноз', np.log(2023) * model.coef_ + model.intercept_)
y_pred = math.e ** model.predict([[np.log(2023)]])
print('ПРОГНОЗ НА ЗИМУ:', y_pred,)

fig, ax = plt.subplots()
x = [2017,2018,2019,2020,2021,2022,2023]
y = [63.400, 67.838, 69.602,75.866,79.659,87.625, '91.528']
ax.set_xlabel('Года')
ax.set_ylabel('Доход')
ax.plot(x, y)
plt.show()
print('------------------------------')



x = np.array([2017,2018,2019,2020,2021,2022]).reshape((-1, 1))
y = np.array([67.527, 72.254, 74.132,80.804,84.844,93.329])
x_log = np.log(x)
y_log = np.log(y)
model = LinearRegression()
model.fit(x_log, y_log)
rez = model.score(x_log,y_log)
print('Коэффициент детерминации:', rez)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
print('логарифмируемый прогноз', np.log(2023) * model.coef_ + model.intercept_)
y_pred = math.e ** model.predict([[np.log(2023)]])
print('ПРОГНОЗ НА ВЕСНУ:', y_pred,)

fig, ax = plt.subplots()
x = [2017,2018,2019,2020,2021,2022,2023]
y = [67.527, 72.254, 74.132,80.804,84.844,93.329, '97.486']
ax.set_xlabel('Года')
ax.set_ylabel('Доход')
ax.plot(x, y)
plt.show()
print('------------------------------')



x = np.array([2017,2018,2019,2020,2021,2022]).reshape((-1, 1))
y = np.array([101.313, 108.405, 111.224,121.234,127.296,140.025])
x_log = np.log(x)
y_log = np.log(y)
model = LinearRegression()
model.fit(x_log, y_log)
rez = model.score(x_log,y_log)
print('Коэффициент детерминации:', rez)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
print('логарифмируемый прогноз', np.log(2023) * model.coef_ + model.intercept_)
y_pred = math.e ** model.predict([[np.log(2023)]])
print('ПРОГНОЗ НА ЛЕТО:', y_pred,)

fig, ax = plt.subplots()
x = [2017,2018,2019,2020,2021,2022,2023]
y = [101.313, 108.405, 111.224,121.234,127.296,140.025, '146.263']
ax.set_xlabel('Года')
ax.set_ylabel('Доход')
ax.plot(x, y)
plt.show()
print('------------------------------')




x = np.array([2017,2018,2019,2020,2021,2022]).reshape((-1, 1))
y = np.array([77.310, 82.722, 84.872,92.511,97.137,106.850])
x_log = np.log(x)
y_log = np.log(y)
model = LinearRegression()
model.fit(x_log, y_log)
rez = model.score(x_log,y_log)
print('Коэффициент детерминации:', rez)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
print('логарифмируемый прогноз', np.log(2023) * model.coef_ + model.intercept_)
y_pred = math.e ** model.predict([[np.log(2023)]])
print('ПРОГНОЗ НА ОСЕНЬ:', y_pred,)

fig, ax = plt.subplots()
x = [2017,2018,2019,2020,2021,2022,2023]
y = [77.310, 82.722, 84.872,92.511,97.137,106.850, '111.610']
ax.set_xlabel('Года')
ax.set_ylabel('Доход')
ax.plot(x, y)
plt.show()
print('------------------------------------------------------------------------------------------------')




print('---2024 YEAR---')

x = np.array([2017, 2018, 2019, 2020, 2021, 2022, 2023]).reshape((-1, 1))
y = np.array([63.400, 67.838, 69.602,75.866, 79.659, 87.625, 95.418])
x_log = np.log(x)
y_log = np.log(y)
model = LinearRegression()
model.fit(x_log, y_log)
rez = model.score(x_log,y_log)
print('Коэффициент детерминации:', rez)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
print('логарифмируемый прогноз', np.log(2024) * model.coef_ + model.intercept_)
y_pred = math.e ** model.predict([[np.log(2024)]])
print('ПРОГНОЗ НА ЗИМУ:', y_pred,)

fig, ax = plt.subplots()
x = [2017,2018,2019,2020,2021,2022,2023,2024]
y = [63.400, 67.838, 69.602,75.866, 79.659, 87.625, 95.418, '99.757']
ax.set_xlabel('Года')
ax.set_ylabel('Доход')
ax.plot(x, y)
plt.show()
print('------------------------------')



x = np.array([2017,2018,2019,2020,2021,2022,2023]).reshape((-1, 1))
y = np.array([67.527, 72.254, 74.132,80.804,84.844,93.329,101.629])
x_log = np.log(x)
y_log = np.log(y)
model = LinearRegression()
model.fit(x_log, y_log)
rez = model.score(x_log,y_log)
print('Коэффициент детерминации:', rez)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
print('логарифмируемый прогноз', np.log(2024) * model.coef_ + model.intercept_)
y_pred = math.e ** model.predict([[np.log(2024)]])
print('ПРОГНОЗ НА ВЕСНУ:', y_pred,)

fig, ax = plt.subplots()
x = [2017,2018,2019,2020,2021,2022,2023,2024]
y = [67.527, 72.254, 74.132,80.804,84.844,93.329,101.629, '106.251']
ax.set_xlabel('Года')
ax.set_ylabel('Доход')
ax.plot(x, y)
plt.show()
print('------------------------------')



x = np.array([2017,2018,2019,2020,2021,2022,2023]).reshape((-1, 1))
y = np.array([101.313, 108.405, 111.224,121.234,127.296,140.025,152.478])
x_log = np.log(x)
y_log = np.log(y)
model = LinearRegression()
model.fit(x_log, y_log)
rez = model.score(x_log,y_log)
print('Коэффициент детерминации:', rez)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
print('логарифмируемый прогноз', np.log(2024) * model.coef_ + model.intercept_)
y_pred = math.e ** model.predict([[np.log(2024)]])
print('ПРОГНОЗ НА ЛЕТО:', y_pred,)

fig, ax = plt.subplots()
x = [2017,2018,2019,2020,2021,2022,2023,2024]
y = [101.313, 108.405, 111.224,121.234,127.296,140.025,152.478, '159.413']
ax.set_xlabel('Года')
ax.set_ylabel('Доход')
ax.plot(x, y)
plt.show()
print('------------------------------')



x = np.array([2017,2018,2019,2020,2021,2022,2023]).reshape((-1, 1))
y = np.array([77.310, 82.722, 84.872,92.511,97.137,106.850,116.353])
x_log = np.log(x)
y_log = np.log(y)
model = LinearRegression()
model.fit(x_log, y_log)
rez = model.score(x_log,y_log)
print('Коэффициент детерминации:', rez)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
print('логарифмируемый прогноз', np.log(2024) * model.coef_ + model.intercept_)
y_pred = math.e ** model.predict([[np.log(2024)]])
print('ПРОГНОЗ НА ОСЕНЬ:', y_pred,)

fig, ax = plt.subplots()
x = [2017,2018,2019,2020,2021,2022,2023,2024]
y = [77.310, 82.722, 84.872,92.511,97.137,106.850,116.353, '121.645']
ax.set_xlabel('Года')
ax.set_ylabel('Доход')
ax.plot(x, y)
plt.show()