import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# **Geometri Parametreleri**
Lx, Ly = 4.0, 2.0  # Dikdörtgen boyutları
circle_center = (Lx / 2, Ly / 2)  # Dairenin merkezi
circle_radius = 0.25  # Dairenin yarıçapı

# **Dairenin Kenarında Başlangıç Mesh'i**
num_boundary_points = 50  # Dairenin etrafındaki başlangıç noktaları
theta = np.linspace(0, 2 * np.pi, num_boundary_points, endpoint=False)
x_boundary = circle_center[0] + circle_radius * np.cos(theta)
y_boundary = circle_center[1] + circle_radius * np.sin(theta)

# **Daire İçindeki Noktaları Filtrele**
def filter_points_inside_circle(points, circle_center, circle_radius):
    # Noktaların daire merkezine olan mesafesini hesapla
    distances = np.linalg.norm(points - circle_center, axis=1)

    # Daire içinde kalan noktaları çıkar
    filtered_points = points[distances >= circle_radius]

    return filtered_points

# **Advancing Front ile İçeri Meshleme**
num_layers = 15  # Katman sayısı
first_layer_thickness = 0.05  # İlk katman kalınlığı
thickness_factor = 1.2  # Katman kalınlık faktörü, her katmanda ne kadar genişleyeceğini belirler

all_x, all_y = list(x_boundary), list(y_boundary)

for i in range(1, num_layers + 1):
    # Her katman için kalınlık hesapla
    new_thickness = first_layer_thickness * (thickness_factor ** (i - 1))  # i-1 çünkü ilk katman zaten verilmiş durumda

    # Yeni katmanın yarıçapını hesapla
    new_radius = circle_radius + new_thickness

    # Yeni katmanın noktalarını hesapla
    x_layer = circle_center[0] + new_radius * np.cos(theta)
    y_layer = circle_center[1] + new_radius * np.sin(theta)

    all_x.extend(x_layer)
    all_y.extend(y_layer)

# **Boundary Layer İçin Mesh Yoğunlaştırma**
boundary_x = np.linspace(0, Lx, 20)
boundary_y = np.linspace(0, Ly, 10)
grid_X, grid_Y = np.meshgrid(boundary_x, boundary_y)
grid_points = np.vstack([grid_X.ravel(), grid_Y.ravel()]).T

# **Tüm Noktaları Birleştir**
points = np.vstack([np.array([all_x, all_y]).T, grid_points])


# **Dairenin İçindeki Noktaları Filtrele**
def filter_points_inside_circle(points, circle_center, circle_radius):
    # Noktaların daire merkezine olan mesafesini hesapla
    distances = np.linalg.norm(points - circle_center, axis=1)

    # Daire içinde kalan noktaları çıkar
    filtered_points = points[distances >= circle_radius]

    return filtered_points


# **Mesh Noktalarını Filtrele**
filtered_points = filter_points_inside_circle(points, circle_center, circle_radius)


# **Delaunay Triangulation ile Mesh Oluştur**
def generate_triangulation(points):
    return Delaunay(points)


tri = generate_triangulation(points)


# **Kalite Metriği Hesaplama (Skewness & Orthogonality)**
def calculate_quality(tri, points):
    skewness = []
    orthogonality = []

    for simplex in tri.simplices:
        pts = points[simplex]
        edges = [np.linalg.norm(pts[i] - pts[(i + 1) % 3]) for i in range(3)]
        max_edge = max(edges)
        min_edge = min(edges)
        skew = min_edge / max_edge  # Skewness ölçüsü
        skewness.append(skew)

        # Orthogonality ölçümü için en küçük iç açıyı hesapla
        angles = []
        for i in range(3):
            vec1 = pts[(i + 1) % 3] - pts[i]
            vec2 = pts[(i + 2) % 3] - pts[i]
            dot_product = np.dot(vec1, vec2)
            norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            angle = np.arccos(dot_product / norm_product) * (180 / np.pi)
            angles.append(angle)
        orthogonality.append(min(angles))

    return np.array(skewness), np.array(orthogonality)


skewness, orthogonality = calculate_quality(tri, points)
threshold_skew = 0.3  # Skewness için eşik değer
target_orthogonality = 40  # En düşük açı threshold
to_refine = np.where((skewness < threshold_skew) | (orthogonality < target_orthogonality))[0]


# **Refinement (Kötü Kaliteli Mesh'leri İyileştirme)**
def refine_mesh(tri, points, to_refine):
    new_points = list(points)
    for idx in to_refine:
        simplex = tri.simplices[idx]
        centroid = np.mean(points[simplex], axis=0)
        new_points.append(centroid)  # Yeni noktayı ekle
    return np.array(new_points)


# Refinement işlemi
enhanced_points = refine_mesh(tri, points, to_refine)
tri = generate_triangulation(enhanced_points)
skewness, orthogonality = calculate_quality(tri, enhanced_points)

# **Mesh Görselleştirme (Skewness Colormap ile)**
fig, ax = plt.subplots(figsize=(12, 6), dpi=(500))
quality_map = np.minimum(skewness, orthogonality / 90)  # Normalize edilerek tek colormap
ax.tripcolor(enhanced_points[:, 0], enhanced_points[:, 1], tri.simplices, facecolors=quality_map, cmap='coolwarm',
             edgecolors='k', linewidth=0.5)
ax.scatter(enhanced_points[:, 0], enhanced_points[:, 1], color="red", s=5, label="Mesh Noktaları")

# **Daireyi Çiz**
circle = plt.Circle(circle_center, circle_radius, color='blue', fill=True, alpha=0.3, label="Dead Zone")
ax.add_patch(circle)

# **Hücre Sınırlarını Çiz**
for simplex in tri.simplices:
    # Üçgenin köşelerinin koordinatlarını al
    pts = enhanced_points[simplex]

    # Üçgenin kenarlarını çiz
    for i in range(3):
        x_vals = [pts[i, 0], pts[(i + 1) % 3, 0]]
        y_vals = [pts[i, 1], pts[(i + 1) % 3, 1]]
        ax.plot(x_vals, y_vals, color='black', lw=0.8)  # Kenarları siyah renkte çiz

ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_xlabel("x ekseni")
ax.set_ylabel("y ekseni")
ax.set_title("Advancing Front ile Refinement'lı Mesh")
ax.legend(loc='upper right')
ax.set_aspect("equal")

plt.colorbar(
    ax.tripcolor(enhanced_points[:, 0], enhanced_points[:, 1], tri.simplices, facecolors=quality_map, cmap='coolwarm'))
plt.show()
