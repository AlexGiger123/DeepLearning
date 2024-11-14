import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Генерація даних
random_gen = np.random.RandomState(1)
x_values = np.linspace(0, 6, 200)
y_values = np.sin(x_values) + np.sin(6 * x_values) + random_gen.normal(0, 0.1, x_values.shape[0])

# Розбиття на тренувальний та тестовий набори
x_train, x_test, y_train, y_test = train_test_split(x_values.reshape(-1, 1), y_values, test_size=0.3, random_state=42)

# Пошук оптимальних параметрів для моделі MLPRegressor
hyperparameters = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam', 'lbfgs'],
    'learning_rate_init': [0.001, 0.01, 0.005],
    'max_iter': [500, 1000]
}

# Ініціалізація та підбір найкращих параметрів
mlp_model = MLPRegressor(random_state=42)
grid_search = GridSearchCV(estimator=mlp_model, param_grid=hyperparameters, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(x_train, y_train)

# Отримання найкращих параметрів
best_params = grid_search.best_params_
print(f'Найкращі гіперпараметри: {best_params}')

# Створення моделі з оптимальними параметрами та збільшеною кількістю ітерацій
final_model = MLPRegressor(
    hidden_layer_sizes=best_params['hidden_layer_sizes'],
    activation=best_params['activation'],
    solver=best_params['solver'],
    learning_rate_init=best_params['learning_rate_init'],
    max_iter=2000,
    random_state=42
)

final_model.fit(x_train, y_train)

# Прогнозування для тренувальних і тестових даних
train_predictions = final_model.predict(x_train)
test_predictions = final_model.predict(x_test)

# Оцінка якості моделі на основі MSE та R2
train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

print(f'Помилка на тренувальному наборі (MSE): {train_mse}, R2: {train_r2}')
print(f'Помилка на тестовому наборі (MSE): {test_mse}, R2: {test_r2}')

# Візуалізація результатів
plt.figure(figsize=(12, 6))

# Графік тренувальних даних та їх прогнозів
plt.subplot(1, 2, 1)
plt.scatter(x_train, y_train, color='purple', label='Train Data')
plt.plot(x_train, train_predictions, color='orange', label='Prediction')
plt.title('Тренувальні дані та Прогноз')
plt.legend()

# Графік тестових даних та їх прогнозів
plt.subplot(1, 2, 2)
plt.scatter(x_test, y_test, color='blue', label='Test Data')
plt.plot(x_test, test_predictions, color='red', label='Prediction')
plt.title('Тестові дані та Прогноз')
plt.legend()

plt.tight_layout()
plt.show()
