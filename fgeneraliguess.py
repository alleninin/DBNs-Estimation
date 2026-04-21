import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import minimize

np.random.seed(42)

def simulate_general_graph_step(adj_matrix, true_alphas, nonlinear_func, sigma=0.01, N=1000):
    n_nodes = len(adj_matrix)
    X_t0 = np.random.normal(-1, 1, size=(n_nodes, N))
    X_t1 = np.zeros((n_nodes, N))

    for i in range(n_nodes):
        parents = np.where(adj_matrix[:, i] == 1)[0]

        if len(parents) == 0:
            X_t1[i] = np.random.normal(0, 1, N)
        else:
            inputs = [true_alphas[(j, i)] * X_t0[j] for j in parents]
            mu = nonlinear_func(*inputs)
            X_t1[i] = np.random.normal(mu, sigma)

    return X_t0, X_t1

def estimate_general_alphas(adj_matrix, X_t0, X_t1, nonlinear_func, target_edges):
    def loss(alpha_vec):
        loss_val = 0.0
        alpha_map = {edge: alpha_vec[i] for i, edge in enumerate(target_edges)}

        for (j, i) in target_edges:
            parents = np.where(adj_matrix[:, i] == 1)[0]
            inputs = [alpha_map[(p, i)] * X_t0[p] for p in parents]
            mu = nonlinear_func(*inputs)
            loss_val += np.mean((X_t1[i] - mu) ** 2)

        return loss_val

    init_alpha = np.ones(len(target_edges))
    res = minimize(loss, init_alpha, method='BFGS')
    return res.x

def plot_general_alpha_errors(adj_matrix, true_alphas, nonlinear_func, sample_sizes):
    edges = [(i, j) for i in range(len(adj_matrix)) for j in range(len(adj_matrix)) if adj_matrix[i, j] == 1]
    errors = []

    for N in sample_sizes:
        X_t0, X_t1 = simulate_general_graph_step(adj_matrix, true_alphas, nonlinear_func, N=N)
        alpha_hat_vec = estimate_general_alphas(adj_matrix, X_t0, X_t1, nonlinear_func, edges)
        true_alpha_vec = np.array([true_alphas[edge] for edge in edges])
        error = np.linalg.norm(alpha_hat_vec - true_alpha_vec)
        print(f"N={N}, Error norm={error:.4f}")
        errors.append(error)

    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, errors, marker='o', label='||α̂ - α||')
    plt.xlabel('Sample Size (N)')
    plt.ylabel('Alphas Error Norm')
    plt.title('Nonlinear Alpha Estimation Error (General Function)')
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()
def plot_graph_structure(adj_matrix):
    G = nx.DiGraph()
    n = len(adj_matrix)
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] == 1:
                G.add_edge(i, j)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=700, node_color='lightblue', edge_color='gray')
    plt.title("Graph Structure (DAG)")
    plt.show()

if __name__ == "__main__":
    n = 10
    adj_matrix = np.zeros((n, n), dtype=int)

    for i in range(1, n):
        adj_matrix[i - 1, i] = 1
        if i >= 2:
            adj_matrix[i - 2, i] = 1
        if i>=3:
            adj_matrix[i-3,i]=1
    print(adj_matrix)
    plot_graph_structure(adj_matrix)

    edges = [(i, j) for i in range(n) for j in range(n) if adj_matrix[i, j] == 1]
    true_alphas = {edge: np.random.uniform(0.5, 2.0) for edge in edges}

    def nonlinear_func(*args):
        return np.cos(np.sum(args, axis=0))

    sample_sizes = [10, 20, 40, 80, 160, 320, 500, 1000, 2000, 5000, 10000]

    plot_general_alpha_errors(adj_matrix, true_alphas, nonlinear_func, sample_sizes)
