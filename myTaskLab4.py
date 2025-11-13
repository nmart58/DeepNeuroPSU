
import torch                    
import torch.nn as nn            
import numpy as np                
import pandas as pd               
import matplotlib.pyplot as plt   


df = pd.read_csv("dataset_simple.csv")  
print(df.head())  # Смотрим первые 5 строк


# ===== Нормализация признаков =====
# Приводим возраст и доход к диапазону [0,1], чтобы обучение было стабильным
df['age_norm'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
df['income_norm'] = (df['income'] - df['income'].min()) / (df['income'].max() - df['income'].min())


# ===== Преобразование данных в тензоры PyTorch =====
# Входные признаки (X) — нормализованные возраст и доход
X = torch.Tensor(df[['age_norm', 'income_norm']].values)
# Целевой признак (y) — метка “купит / не купит”
y = torch.Tensor(df[['will_buy']].values)


# ===== Определение архитектуры нейросети =====
class BuyerClassifier(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # Последовательность слоёв: вход → скрытый слой → выход
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),  # Линейный слой: 2 входа → 8 нейронов
            nn.ReLU(),                        # Активация ReLU добавляет нелинейность
            nn.Linear(hidden_size, hidden_size), # Ещё один скрытый слой
            nn.ReLU(),
            nn.Linear(hidden_size, out_size), # Выходной слой: 8 нейронов → 1 выход
            nn.Sigmoid()                      # Сигмоида: превращает выход в вероятность [0,1]
        )

    def forward(self, x):
        # Прямое распространение данных по слоям
        return self.layers(x)


# ===== Создание экземпляра модели =====
inputSize = X.shape[1]     # Количество входных признаков (2)
hiddenSize = 8             # Количество нейронов в скрытом слое
outputSize = 1             # Один выход (вероятность "купит")

net = BuyerClassifier(inputSize, hiddenSize, outputSize)  # Создаём сеть


# ===== Настройка функции потерь и оптимизатора =====
lossFn = nn.BCELoss()                          # Функция потерь — бинарная кросс-энтропия
#Сравнивает предсказанную вероятность с реальным значением (0 или 1) и выдаёт 
#число — насколько сильно сеть ошибается
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # Оптимизатор Adam со скоростью обучения 0.01


# ===== 8. Цикл обучения =====
epochs = 5000  # Количество итераций обучения

for i in range(epochs):
    pred = net(X)            # Прямой проход (вычисляем предсказания)
    loss = lossFn(pred, y)   # Вычисляем ошибку между предсказаниями и реальными метками

    optimizer.zero_grad()    # Обнуляем старые градиенты
    loss.backward()          # Обратное распространение ошибки (градиенты)
    optimizer.step()         # Обновляем веса сети

    # Каждые 500 эпох выводим текущую ошибку
    if i % 500 == 0:
        print(f'Эпоха {i}: ошибка = {loss.item():.6f}')


# ===== Оценка модели после обучения =====
with torch.no_grad():                 # Отключаем вычисление градиентов (только предсказание)
    preds = net(X)                    # Считаем вероятности
    preds_class = (preds >= 0.5).float()  # Преобразуем в классы (0 или 1) по порогу 0.5


# ===== 10. Подсчёт ошибок и точности =====
errors = torch.sum(torch.abs(y - preds_class))  # Считаем количество несовпадений
accuracy = 100 * (1 - errors.item() / len(y))   # Вычисляем точность в процентах

print(f'\nКоличество ошибок: {int(errors.item())} из {len(y)} ({accuracy:.1f}% точность)')


# ===== 11. Визуализация результатов классификации с подписями =====
plt.figure(figsize=(6,5))

# Цвет точек показывает предсказанный класс:
# 0 = не купит (синий), 1 = купит (красный)
scatter = plt.scatter(
    df['age'], 
    df['income'], 
    c=preds_class.squeeze(), 
    cmap='bwr',  # blue-red colormap
    s=70,
    edgecolors='k'  # добавляем черную границу точкам для наглядности
)

# Добавляем подписи к осям
plt.xlabel('Возраст (years)', fontsize=12)
plt.ylabel('Доход (тыс. руб.)', fontsize=12)
plt.title('Классификация покупателей: купит / не купит', fontsize=14)

# Добавляем легенду вручную
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Не купит', markerfacecolor='blue', markersize=10, markeredgecolor='k'),
    Line2D([0], [0], marker='o', color='w', label='Купит', markerfacecolor='red', markersize=10, markeredgecolor='k')
]
plt.legend(handles=legend_elements, loc='upper left')

# Показываем график
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
