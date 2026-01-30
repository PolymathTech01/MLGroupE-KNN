import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


points = {
    "blue": [[2, 4, 2], [3, 6, 1], [8, 7, 3], [5, 3, 2], [7, 2, 4]],
    "red": [[1, 7, 3], [4, 5, 2], [6, 4, 1], [7, 5, 3], [4, 2, 2]]
}
new_point = [5, 5, 2]


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))


class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.point = None

    def fit(self, points):
        self.points = points

    def predict(self, new_point):
        distances = []

        for category in self.points:
            for point in self.points[category]:
                distance = euclidean_distance(point, new_point)
                distances.append((distance, category))

        categories = [category for _, category in sorted(distances)[:self.k]]
        result = Counter(categories).most_common(1)[0][0]
        return result


knn = KNearestNeighbors(k=3)
knn.fit(points)
prediction = knn.predict(new_point)
print(f"The predicted category for the new point {new_point} is: {prediction}")


fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(projection='3d')
ax.grid(True, color="#CCCCCC")
ax.figure.set_facecolor("#F5F5F5")
ax.tick_params(axis="x", color="white")
ax.tick_params(axis="y", color="white")

for point in points["blue"]:
    ax.scatter(point[0], point[1], point[2], color="blue", s=100)
for point in points["red"]:
    ax.scatter(point[0], point[1], point[2], color="red", s=100)
color = "blue" if prediction == "blue" else "red"
ax.scatter(new_point[0], new_point[1], new_point[2], color=color,
           s=200, marker="*", zorder=100)

for point in points["blue"]:

    ax.plot(
        [new_point[0], point[0]],
        [new_point[1], point[1]],
        [new_point[2], point[2]],
        color="blue", linestyle="--", linewidth=0.5)

for point in points["red"]:
    ax.plot(
        [new_point[0], point[0]],
        [new_point[1], point[1]],
        [new_point[2], point[2]],
        color="red", linestyle="--", linewidth=0.5)
plt.show()
