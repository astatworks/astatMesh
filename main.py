import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# **Geometri Parametreleri**
Lx, Ly = 2.0, 1  # Dikdörtgen boyutları
circle_center = (Lx / 2, Ly / 2)  # Dairenin merkezi
circle_radius = 0.25  # Dairenin yarıçapı

# **Daire Sınırını Oluştur (Başlangıç Noktaları)**
num_boundary_points = 25  # Dairenin etrafındaki başlangıç noktaları
theta = np.linspace(0, 2 * np.pi, num_boundary_points, endpoint=False)
x_boundary = circle_center[0] + circle_radius * np.cos(theta)
y_boundary = circle_center[1] + circle_radius * np.sin(theta)

# **Mesh Alanına Yeni Noktalar Ekleyerek İçeri Doğru Genişlet**
num_layers = 5  # Kaç katman ekleyeceğimiz
layer_thickness = 0.02  # Her katmanın mesafesi

all_x, all_y = list(x_boundary), list(y_boundary)

for i in range(1, num_layers + 1):
    new_radius = circle_radius + i * layer_thickness  # Yeni katmanın yarıçapı
    x_layer = circle_center[0] + new_radius * np.cos(theta)
    y_layer = circle_center[1] + new_radius * np.sin(theta)
    all_x.extend(x_layer)
    all_y.extend(y_layer)

# **Dikdörtgenin Kenarlarına Noktalar Ekle**
grid_x = np.linspace(0, Lx, 25)
grid_y = np.linspace(0, Ly, 12)
grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
grid_points = np.vstack([grid_X.ravel(), grid_Y.ravel()]).T

# **Tüm Noktaları Birleştir**
points = np.vstack([np.array([all_x, all_y]).T, grid_points])

# **Delaunay Triangulation ile Unstructured Mesh Oluştur**
tri = Delaunay(points, incremental=False)

# **Mesh Görselleştirme**
fig, ax = plt.subplots(figsize=(6, 3))
ax.triplot(points[:, 0], points[:, 1], tri.simplices, color="k", lw=0.5)
ax.scatter(points[:, 0], points[:, 1], color="red", s=5, label="Mesh Noktaları")

# **Daireyi Çiz**
circle = plt.Circle(circle_center, circle_radius, color='blue', fill=True, alpha=0.3, label="Dead Zone")
ax.add_patch(circle)

ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_xlabel("x ekseni")
ax.set_ylabel("y ekseni")
ax.set_title("Advancing Front Method ile Unstructured Mesh")
ax.legend()
ax.set_aspect("equal")

plt.show()
