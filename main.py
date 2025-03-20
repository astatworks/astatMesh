import numpy as np
import matplotlib.pyplot as plt
import triangle

# Geometri Parametreleri
Lx, Ly = 4.0, 2.0  # Dikdörtgen boyutları
circle_center = (Lx / 2, Ly / 2)  # Daire merkezi
circle_radius = 0.5  # Daire yarıçapı

# Dikdörtgenin Köşe Noktaları
rectangle = np.array([[0, 0], [Lx, 0], [Lx, Ly], [0, Ly]])

# Daire Sınır Noktaları
num_circle_pts = 40  # Daire etrafındaki nokta sayısı
theta = np.linspace(0, 2 * np.pi, num_circle_pts, endpoint=False)
circle_x = circle_center[0] + circle_radius * np.cos(theta)
circle_y = circle_center[1] + circle_radius * np.sin(theta)
circle = np.column_stack((circle_x, circle_y))

# Triangle Kütüphanesi İçin Giriş Formatı
segments = []
points = rectangle.tolist() + circle.tolist()
for i in range(len(rectangle)):
    segments.append([i, (i + 1) % len(rectangle)])

circle_start_idx = len(rectangle)
for i in range(len(circle)):
    segments.append([circle_start_idx + i, circle_start_idx + (i + 1) % len(circle)])

# Meshi Üret (Ruppert's Algorithm ile Refinement)
geodata = {"vertices": np.array(points), "segments": np.array(segments)}
mesh = triangle.triangulate(geodata, "pqa0.02")

# Üçgenlerin İçindeki Noktaları Filtrele
filtered_triangles = []
for tri in mesh["triangles"]:
    tri_pts = mesh["vertices"][tri]
    centroids = np.mean(tri_pts, axis=0)
    dist_to_center = np.linalg.norm(centroids - np.array(circle_center))
    if dist_to_center > circle_radius:
        filtered_triangles.append(tri)
filtered_triangles = np.array(filtered_triangles)

# Mesh Görselleştirme
plt.figure(figsize=(10, 5))
plt.triplot(mesh["vertices"][:, 0], mesh["vertices"][:, 1], filtered_triangles, "k-")
plt.gca().set_aspect("equal")
plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Ruppert's Algorithm ile Dairenin Hariç Bırakıldığı Mesh")
plt.show()
