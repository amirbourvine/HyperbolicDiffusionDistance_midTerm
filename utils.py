import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
from matplotlib import colors
import networkx as nx
from scipy.linalg import eigh
from sklearn.manifold import MDS
from scipy.spatial import distance

# CONSTANTS:

CONST_K = 19
CONST_C = 5
TOL = 1e-6
TREE_LEVELS = 5
ALPHA = 0.5



def generate_full_binary_tree_points():
    width = 1
    depth = 10
    var = 0.05

    num_nodes = 2 ** TREE_LEVELS - 1
    points = []
    rand_factor_x = [random.uniform(1 - var, 1 + var) for _ in range(num_nodes)]
    rand_factor_y = [random.uniform(1 - var, 1 + var) for _ in range(num_nodes)]
    colors_res = []

    for i in range(1, num_nodes + 1):
        level = (i).bit_length()
        position = (((i - (2 ** (level - 1) - 1)) * (1 / (2 **(level - 1)))) - ((1 / (2 **(level - 1))) / 2)) - (width / 2)
        points.append([position * rand_factor_x[i - 1] * width, (level - 1) / TREE_LEVELS * rand_factor_y[i - 1] * depth])
        colors_res.append(level)

    cmap = cm.Set1
    norm = colors.BoundaryNorm(np.arange(0.5, 6, 1), cmap.N)

    return points,colors_res,norm,cmap

def plot_tree():
  points,colors_res,norm,cmap = generate_full_binary_tree_points()

  x_points = [p[0] for p in points]
  y_points = [p[1] for p in points]

  for i in range(1,len(x_points)+1):
    if 2*i-1 < len(x_points):
      plt.arrow(x_points[i-1],y_points[i-1],x_points[2*i-1]-x_points[i-1],y_points[2*i-1]-y_points[i-1],width=0.002,color='black',head_length=0.0,head_width=0.0)
    if 2*i < len(x_points):
      plt.arrow(x_points[i-1],y_points[i-1],x_points[2*i]-x_points[i-1],y_points[2*i]-y_points[i-1],width=0.002,color='black',head_length=0.0,head_width=0.0)

  plt.scatter(x=x_points, y=y_points, marker='o', c=colors_res, norm=norm, cmap = cmap,edgecolors='black',s=100)
  plt.title("Generated Points for Full Binary Tree")
  plt.xlabel("X-coordinate")
  plt.ylabel("Y-coordinate")
  cbar = plt.colorbar(ticks=np.linspace(1, 5, 5))
  cbar.set_label('level')
  plt.show()

  return points,x_points,y_points


def create_graph():
    G = nx.Graph()

    # Create a full binary tree with the specified number of levels
    nodes = 2 ** TREE_LEVELS - 1
    for i in range(1, nodes + 1):
        G.add_node(i)

    for i in range(1, nodes + 1):
        if 2 * i <= nodes:
            G.add_edge(i, 2 * i)
        if 2 * i + 1 <= nodes:
            G.add_edge(i, 2 * i + 1)

    return G


def shortest_paths_matrix():
    G = create_graph()

    result = nx.shortest_path(G)

    nodes_num = G.number_of_nodes()

    res_mat = np.zeros((nodes_num,nodes_num))

    for key,dic in result.items():
      for in_key, path in dic.items():
        res_mat[key-1,in_key-1] = len(path)-1

    return res_mat

def mat_show(mat, cmap):
    plt.matshow(mat, cmap=cmap)
    plt.colorbar()
    plt.show()

def calc_P(d, apply_2_norm=False):
  epsilon = CONST_C*np.median(d)
  W = np.exp(-1*d/epsilon)

  if apply_2_norm:
    S_vec = np.sum(W,axis=1)

    S = np.diag(1/S_vec)
    W_gal = S@W@S

    D_vec = np.sum(W_gal,axis=1)
    D = np.diag(1 / D_vec)
    P = D@W_gal

  else:
    D_vec = np.sum(W,axis=1)
    D = np.diag(1/ D_vec)
    P = D@W

  return P



def svd_symmetric(M):
  s,u = eigh(M)  #eigenvalues and eigenvectors

  u = u[:, np.argsort(s)[::-1]]
  s = (np.sort(s)[::-1])

  v = u.copy()
  v[:,s<0] = -u[:,s<0] #replacing the corresponding columns with negative sign

  s = abs(s)

  s = np.where(s>TOL, s, TOL)

  return u, s, v.T


def calc_svd_p(d):
  epsilon = CONST_C*np.median(d)
  W = np.exp(-1*d/epsilon)
  S_vec = np.sum(W,axis=1)

  S = np.diag(1/S_vec)
  W_gal = S@W@S

  D_vec = np.sum(W_gal,axis=1)

  D_minus_half = np.diag(1 / np.sqrt(D_vec))
  D_plus_half = np.diag(np.sqrt(D_vec))


  M = D_minus_half@W_gal@D_minus_half

  U,S,UT = svd_symmetric(M)

  return  (D_minus_half@U),S,(UT@D_plus_half)


def multi_scale_propagated_densities_colors(shortest_paths_mat,i, k):
    e_i = np.zeros(shortest_paths_mat.shape[0])
    e_i[i] = 1

    U, S, Vt = calc_svd_p(shortest_paths_mat)
    S = np.float_power(S, 2 ** (-k))
    aux = U @ np.diag(S) @ Vt

    values = aux @ e_i

    cmap = cm.Reds
    norm = Normalize(vmin=np.min(values), vmax=np.max(values))

    return values,norm,cmap

def display_figure_B(points,x_points,y_points,shortest_paths_mat):
  # Create subplots
  sigma_i_to_plot = [0, 1, 10, 20]
  sigma_k_to_plot = [0, 1, CONST_K]

  _, axs = plt.subplots(len(sigma_i_to_plot), len(sigma_k_to_plot), figsize=(9, len(points) / 4), sharex=True, sharey=True)
  plt.suptitle("phi(i,k)=P^(2^-k)*ei")
  for i in range(len(sigma_i_to_plot)):
      for k in range(len(sigma_k_to_plot)):
        for jj in range(1,len(x_points)+1):
          if 2*jj-1 < len(x_points):
            axs[i][k].arrow(x_points[jj-1],y_points[jj-1],x_points[2*jj-1]-x_points[jj-1],y_points[2*jj-1]-y_points[jj-1],width=0.0002,color='black',head_length=0.0,head_width=0.0)
          if 2*jj < len(x_points):
            axs[i][k].arrow(x_points[jj-1],y_points[jj-1],x_points[2*jj]-x_points[jj-1],y_points[2*jj]-y_points[jj-1],width=0.0002,color='black',head_length=0.0,head_width=0.0)

        colors,norm,cmap = multi_scale_propagated_densities_colors(shortest_paths_mat, sigma_i_to_plot[i], sigma_k_to_plot[k])
        pcm = axs[i][k].scatter(x_points, y_points, c=colors, norm=norm, cmap=cmap,edgecolors='black',s=60)
        axs[i][k].set_title(f'phi({sigma_i_to_plot[i]}, {sigma_k_to_plot[k]})')

        plt.colorbar(pcm, ax=axs[i][k])


  # Adjust layout
  plt.tight_layout()

  # Show the plot
  plt.show()


#The HDE function
def hde(shortest_paths_mat):
  U, S, Vt = calc_svd_p(shortest_paths_mat)

  X = np.zeros((CONST_K + 1, shortest_paths_mat.shape[0], shortest_paths_mat.shape[1] + 1), dtype=np.complex64)
  S_keep=S
  for k in range (0, CONST_K + 1):
    S = np.float_power(S_keep, 2 ** (-k))

    aux = U @ np.diag(S) @ Vt

    aux = np.transpose(np.sqrt((np.where(aux > TOL, aux, TOL))))
    X[k] = np.transpose(np.concatenate((aux, np.full(shortest_paths_mat.shape[0], 2 ** (k * ALPHA - 2)).reshape(1, -1)), axis=0))

  return X

def display_figure_C(shortest_paths_mat):
    X = hde(shortest_paths_mat)
    X = np.abs(X)
    fig, axs = plt.subplots(1,3)
    fig.tight_layout()
    pos = axs[0].matshow((X[0]), cmap=plt.cm.viridis)
    fig.colorbar(pos, ax=axs[0], shrink = 0.28)
    axs[0].set_title("0")
    pos = axs[1].matshow((X[1]), cmap=plt.cm.viridis)
    fig.colorbar(pos, ax=axs[1], shrink = 0.28)
    axs[1].set_title("1")
    pos = axs[2].matshow((X[CONST_K]), cmap=plt.cm.viridis)
    fig.colorbar(pos, ax=axs[2], shrink = 0.28)
    axs[2].set_title("K")

    return X


#The HDD function
def hdd(X, P):
  d_HDD = np.zeros(P.shape)
  for i in range (P.shape[0]):
    for j in range (P.shape[0]):
      sum = 0
      for k in range(CONST_K+1):
        sum += 2 * np.arcsinh((2 ** (-1 * k * ALPHA + 1)) * np.linalg.norm(X[k][i] - X[k][j]))
      d_HDD[i][j] = sum
  return d_HDD


def display_figure_D(d_HDD):
  embedding = MDS(n_components=2, normalized_stress='auto', dissimilarity='precomputed')
  X_transformed = embedding.fit_transform(d_HDD)

  levels = np.zeros(X_transformed.shape[0])
  for i in range(levels.shape[0]):
    levels[i] = (i+1).bit_length()

  cmap = cm.Set1
  norm = colors.BoundaryNorm(np.arange(0.5, 6, 1), cmap.N)

  plt.scatter(X_transformed[:, 0], X_transformed[:, 1],c=levels, norm=norm, cmap = cmap, edgecolors='black',s=100)
  cbar = plt.colorbar(ticks=np.linspace(1, 5, 5))
  cbar.set_label('level')
  plt.show()



def calculate_P_from_points(x_points, y_points, z_points):
    coordinates = np.column_stack((x_points, y_points, z_points))

    d = np.linalg.norm(coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :], axis=-1)

    return calc_P(d)

def plot_embedding(x_points, y_points, z_points, colors):
  curr_val = []
  curr_vec = []

  eigenvalues, eigenvectors = np.linalg.eig(calculate_P_from_points(x_points, y_points, z_points))
  eigenvaluesKeep = eigenvalues
  eigenvectorsKeep = eigenvectors

  for i in range(eigenvaluesKeep.shape[0]):
    val = eigenvaluesKeep[i]
    vec = eigenvectorsKeep[:,i]
    if np.imag(val)==0:
      curr_val.append(np.real(val))
      curr_vec.append(vec)

  eigenvalues = np.array(curr_val)
  eigenvectors = (curr_vec)
  eigenvalues_indexes = np.argsort(-1*eigenvalues)

  ind1 = eigenvalues_indexes[1]
  ind2 = eigenvalues_indexes[2]


  lambda_1 = eigenvalues[ind1]
  phi_1 = np.real(eigenvectors[ind1])
  v1 = lambda_1*phi_1

  lambda_2 = eigenvalues[ind2]
  phi_2 = np.real(eigenvectors[ind2])
  v2 = lambda_2*phi_2

  _ = plt.figure()
  ax = plt.axes()

  ax.scatter(v1,v2, color= colors)

  plt.show()

# Function to create points based on the given parametric equations
def helix_ring_parametric_equations():
    # Parameters for the parametric equations
    R = 6  # Major radius
    r = 2  # Minor radius
    n =  10 # Parameter for the equation
    points_num = 1500

    t = np.linspace(0, 2 * np.pi, points_num)

    x = (R + r * np.cos(n * t)) * np.cos(t)
    y = (R + r * np.cos(n * t)) * np.sin(t)
    z = r * np.sin(n * t)

    cmap = cm.autumn
    norm = Normalize(vmin=np.min(t), vmax=np.max(t))
    colors = []
    for i in range(t.shape[0]):
        colors.append(cmap(norm(t[i])))

    return x, y, z, colors

def plot_and_embedding(parametric_equations_func):
    x, y, z, colors = parametric_equations_func()
   # Plot the points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=5, color=colors, marker='o')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    plot_embedding(x,y,z, colors)




def tours_parametric_equation():
  # Parameters for the torus
    R = 3  # Major radius
    r = 1  # Minor radius
    theta_resolution = 50  # Number of points around the torus
    phi_resolution = 60  # Number of points along the cross-section of the torus

    # Create arrays for theta and phi angles
    theta = np.linspace(0, 2 * np.pi, theta_resolution)
    phi = np.linspace(0, 2 * np.pi, phi_resolution)

    # Calculate torus coordinates directly
    x_points = np.zeros(theta_resolution * phi_resolution)
    y_points = np.zeros(theta_resolution * phi_resolution)
    z_points = np.zeros(theta_resolution * phi_resolution)

    index = 0
    for t in theta:
        for p in phi:
            x_points[index] = (R + r * np.cos(p)) * np.cos(t)
            y_points[index] = (R + r * np.cos(p)) * np.sin(t)
            z_points[index] = r * np.sin(p)
            index += 1

    cmap = cm.autumn
    norm = Normalize(vmin=np.min(x_points), vmax=np.max(x_points))
    colors = []
    for i in range(x_points.shape[0]):
        colors.append(cmap(norm(x_points[i])))

    return x_points, y_points, z_points, colors


def random_points_on_rectangle(center, width, height, n):
    # Generate n random points within the rectangle
    x = np.random.uniform(center[0] - width / 2, center[0] + width / 2, n)
    y = np.random.uniform(center[1] - height / 2, center[1] + height / 2, n)

    return x, y

def random_points_on_circle(center, radius, n):
    # Generate n random radial distances within the circle
    radial_distances = np.sqrt(np.random.uniform(0, radius**2, n))

    # Generate n random angles
    angles = np.random.uniform(0, 2 * np.pi, n)

    # Calculate coordinates of random points on the circle
    x = center[0] + radial_distances * np.cos(angles)
    y = center[1] + radial_distances * np.sin(angles)

    return x, y

def plot_weights():
    center = 15
    radius = 10
    num_points = 120
    x_circle1, y_circle1 = random_points_on_circle((center, 0), radius, num_points)
    x_circle2, y_circle2 = random_points_on_circle((-center, 0), radius, num_points)

    num_points_rectangle = 15
    epsilon_width = 0.1
    height_radius_ration = 0.3
    x_bridge, y_bridge = random_points_on_rectangle((0, 0), (center - radius) * (2 + epsilon_width), radius * height_radius_ration, num_points_rectangle)

    x_points = np.concatenate((x_circle1, x_circle2, x_bridge))
    y_points = np.concatenate((y_circle1, y_circle2, y_bridge))

    
    first_circle_first_index = 0
    first_circle_last_index = num_points
    index = np.random.randint(first_circle_first_index, first_circle_last_index)

    plot_points_with_eucliean_distance_color(x_points, y_points, index)

    plot_points_with_diffusion_distance_color(x_points, y_points, index)





def plot_points_with_eucliean_distance_color(x_values, y_values, index):
    # Get the coordinates of the point at the specified index
    index_point = (x_values[index], y_values[index])

    # Create numpy arrays for x and y values
    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # Calculate Euclidean distances from the index point to all other points
    distances = np.linalg.norm(np.vstack([x_values - index_point[0], y_values - index_point[1]]).T, axis=1)

    # Normalize distances to range [0, 1]
    normalized_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

    # Define colors based on normalized distances (shades of red)
    colors = 1 - normalized_distances

    # Plot the points with shaded colors
    plt.scatter(x_values, y_values, c=colors, cmap="Reds")

    plt.colorbar()

    # Plot the index point in black
    plt.scatter(index_point[0], index_point[1], color='b', label=f'Target point (#{index})')

    # Set labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points with euclidian Distance-based Colors')
    plt.legend()

    plt.gca().set_aspect('equal')

    plt.show()

def plot_points_with_diffusion_distance_color(x_points, y_points, index):

    # Create numpy arrays for x and y values
    points = np.array([(x_points[i], y_points[i]) for i in range(len(x_points))])

    distances = np.array(distance.cdist(points, points, 'euclidean'))

    colors,norm,cmap = multi_scale_propagated_densities_colors(distances, index, 0)

    # Plot the points with shaded colors
    plt.scatter(x_points, y_points, c=colors, cmap=cmap, norm=norm)

    plt.colorbar()

    # Plot the index point in black
    plt.scatter(x_points[index], y_points[index], color='b', label=f'Target point (#{index})')

    # Set labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points with diffusion Distance-based Colors')
    plt.legend()

    plt.gca().set_aspect('equal')

    plt.show()