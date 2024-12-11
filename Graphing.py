import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import scipy

# Data for the baseline cost (y=mx) for the 4o model
x = np.linspace(0, 10, 100)  # Number of reasoning steps
baseline_cost = 2 * x  # Linear growth

# Data for the system's computational cost (exponential growth)
system_cost = np.exp(0.6 * x)  # Exponential growth representing skyrocketing costs

# Data for cost with specifically trained helper models (reduced growth)
reduced_cost = system_cost / 1.5  # Adjusted to show reduced computational load

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, baseline_cost, label="Baseline Model", linestyle='--', color='blue')
plt.plot(x, system_cost, label="Single Model Iterative", color='red')
plt.plot(x, reduced_cost, label="System of Models", color='green')

# Add labels, title, and legend
plt.title("Computational Costs vs Queries", fontsize=14)
plt.xlabel("Prompt Complexity", fontsize=12)
plt.ylabel("Computational Cost", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

# Hide axis numbers but keep axis titles
plt.gca().xaxis.set_ticks([])  # Hide x-axis tick numbers
plt.gca().yaxis.set_ticks([])  # Hide y-axis tick numbers

plt.tight_layout()

# Show the plot
plt.show()


# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, baseline_cost, label="Baseline Model", linestyle='--', color='blue')
plt.plot(x, system_cost, label="Single Model Iterative", color='red')
plt.plot(x, reduced_cost, label="System of Models", color='green')

# Add labels, title, and legend
plt.title("Computational Costs (Log Scale) vs Queries", fontsize=14)
plt.xlabel("Prompt Complexity", fontsize=12)
plt.ylabel("Computational Cost (Log Scale)", fontsize=12)
plt.legend(fontsize=12)

# Set the y-axis to log scale
plt.yscale('log')

# Add grid and other formatting
#plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.gca().xaxis.set_ticks([])  # Hide x-axis tick numbers
plt.gca().yaxis.set_ticks([])  # Hide y-axis tick numbers

plt.tight_layout()

# Show the plot
plt.show()


exit(1)

def generate_3d_heatmap_background(ax, pos, heat_values, resolution=50):
    """
    Generates a 3D heatmap based on node positions and heat values.
    """
    x = np.array([p[0] for p in pos.values()], dtype=float)
    y = np.array([p[1] for p in pos.values()], dtype=float)  # Explicitly cast to float
    z = np.array([p[2] for p in pos.values()], dtype=float)
    heat = np.array([heat_values[node] for node in pos.keys()])

    # Add small noise to avoid degenerate dimensions
    y += np.random.normal(0, 1e-3, size=y.shape)  # Add noise to y-coordinates

    # Create a grid for interpolation
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    zi = np.linspace(z.min(), z.max(), resolution)
    xi, yi, zi = np.meshgrid(xi, yi, zi)

    # Interpolate heatmap in 3D space
    from scipy.interpolate import griddata
    heat_interpolated = griddata((x, y, z), heat, (xi, yi, zi), method='linear', fill_value=np.min(heat))

    # Plot the heatmap as a scatter plot
    ax.scatter(xi.flatten(), yi.flatten(), zi.flatten(), c=heat_interpolated.flatten(), cmap='plasma', alpha=0.2, s=1)
# Create a larger tree graph
G = nx.DiGraph()
nodes = ['Root']
edges = []

# Define a deeper and more complex tree structure
branches = {
    "Root": [f"Branch{i}" for i in range(1, 6)],
    **{f"Branch{i}": [f"Leaf{i}_{j}" for j in range(1, 4)] for i in range(1, 6)},
    **{f"Leaf{i}_{j}": [f"EndNode{i}_{j}_{k}" for k in range(1, 3)] for i in range(1, 6) for j in range(1, 4)},
}

# Add nodes and edges
for parent, children in branches.items():
    nodes.extend(children)
    edges.extend([(parent, child) for child in children])

G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Identify end nodes
end_nodes = [node for node in G.nodes if "EndNode" in node]

# Generate heatmap values for nodes
np.random.seed(42)
heat_values = {node: np.random.rand() for node in G.nodes}

# Generate tree-like 3D positions
def tree_layout(graph, scale=1.0, levels=5):
    pos = {}
    root_nodes = ["Root"]
    z_step = -scale / levels
    current_level_nodes = root_nodes
    z = 0
    while current_level_nodes:
        x_spacing = scale / len(current_level_nodes)
        for i, node in enumerate(current_level_nodes):
            pos[node] = (i * x_spacing - scale / 2, 0, z)
        z += z_step
        current_level_nodes = [
            child for node in current_level_nodes for child in graph.successors(node)
        ]
    return pos

pos = tree_layout(G, scale=5, levels=5)

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot 3D heatmap background
generate_3d_heatmap_background(ax, pos, heat_values)

# Plot edges
for edge in G.edges:
    x_vals = [pos[edge[0]][0], pos[edge[1]][0]]
    y_vals = [pos[edge[0]][1], pos[edge[1]][1]]
    z_vals = [pos[edge[0]][2], pos[edge[1]][2]]
    ax.plot(x_vals, y_vals, z_vals, color='black', alpha=0.6)

# Plot nodes
ax.scatter(
    [pos[node][0] for node in G.nodes if node not in end_nodes],
    [pos[node][1] for node in G.nodes if node not in end_nodes],
    [pos[node][2] for node in G.nodes if node not in end_nodes],
    c=[heat_values[node] for node in G.nodes if node not in end_nodes],
    cmap='plasma',
    s=100,
    edgecolor='black'
)

# Highlight end nodes
ax.scatter(
    [pos[node][0] for node in end_nodes],
    [pos[node][1] for node in end_nodes],
    [pos[node][2] for node in end_nodes],
    c='red',
    s=200,
    edgecolor='black',
    label='End Nodes'
)

# Add labels for nodes
for node, (x, y, z) in pos.items():
    ax.text(x, y, z, node, fontsize=6, ha='center', va='center')

# Final touches
ax.set_title("3D Tree-Like Reasoning Pathways with End Nodes and Heatmap", fontsize=14)
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()