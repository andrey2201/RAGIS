import pandas as pd
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances
import itertools

# Зареждане на CSV файла с локации
df = pd.read_csv("data/locations.csv")

# Проверка на колоните
required_columns = {'name', 'lat', 'lon'}
if not required_columns.issubset(set(df.columns)):
    raise ValueError(f"CSV файлът трябва да съдържа следните колони: {required_columns}")

# Извличане на координати и имена
names = df['name'].tolist()
coords = list(zip(df['lat'], df['lon']))


# Изчисляване на разстояния между всички двойки локации

distances = []
for (i, coord1), (j, coord2) in itertools.combinations(enumerate(coords), 2):
    d = geodesic(coord1, coord2).meters
    distances.append({
        'from': names[i],
        'to': names[j],
        'distance_m': d
    })

dist_df = pd.DataFrame(distances)

# --- Печат на статистика ---
print("\n Статистика на разстоянията между всички двойки локации:\n")
print(dist_df['distance_m'].describe())

# --- Хистограма ---
plt.figure(figsize=(10, 6))
sns.histplot(dist_df['distance_m'], bins=20, kde=True, color="skyblue")
plt.title('Разпределение на разстоянията между локациите (в метри)')
plt.xlabel('Разстояние (м)')
plt.ylabel('Честота')
plt.grid(True)
plt.tight_layout()
plt.show()

# Клъстерен анализ с DBSCAN

# Преобразуване в радиани за Haversine
coords_rad = np.radians(df[['lat', 'lon']].values)

# DBSCAN на база Haversine
kms_per_radian = 6371.0088
epsilon = 0.5 / kms_per_radian  # 500 метра

db = DBSCAN(eps=epsilon, min_samples=2, metric='haversine').fit(coords_rad)
df['cluster'] = db.labels_

# Визуализация на клъстери

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='lon', y='lat', hue='cluster', palette='tab10', data=df, s=100
)
plt.title('Клъстерен анализ на локациите (DBSCAN)')
plt.xlabel('Дължина')
plt.ylabel('Ширина')
plt.grid(True)
plt.tight_layout()
plt.show()
