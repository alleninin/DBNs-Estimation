import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(12)
def simulate_graph_step(adj_matrix, true_alphas, initial_values, c=0.0, sigma=0.01, N=1000):
    n_nodes = len(adj_matrix)
    X_t0 = np.random.normal(50, 1, size=(n_nodes, N))
    X_t1 = np.zeros((n_nodes, N))
    for i in range(n_nodes):
        parents = np.where(adj_matrix[:, i] == 1)[0]
        if len(parents) == 0:
            X_t1[i] = np.random.normal(0, 1, N)
        else:
            mu = np.zeros(N)
            for j in parents:
                mu += true_alphas[j] * X_t0[j]
            mu = mu + c
            X_t1[i] = np.random.normal(mu, sigma)
    return X_t0, X_t1

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
    nx.draw(G, pos, with_labels=True, arrows=True,
            node_size=700, node_color='lightblue', edge_color='gray')
    plt.title("Graph Structure (DAG)")
    plt.show()

def estimate_alphas_mle(adj_matrix, X_t0, X_t1, target_node):
    parents = np.where(adj_matrix[:, target_node] == 1)[0]
    if len(parents) == 0:
        return parents, np.array([])
    x = X_t1[target_node]
    U = X_t0[parents]
    k = len(parents)
    A = np.zeros((k, k))
    b = np.zeros(k)
    for i in range(k):
        b[i] = np.mean(x * U[i])
        for j in range(k):
            A[i, j] = np.mean(U[j] * U[i])
    alpha_hat = np.linalg.solve(A, b)
    return parents, alpha_hat

def estimate_alphas_map(adj_matrix, X_t0, X_t1, target_node, sigma=0.01, s_prior=1.0):
    parents = np.where(adj_matrix[:, target_node] == 1)[0]
    if len(parents) == 0:
        return parents, np.array([])
    x = X_t1[target_node]
    U = X_t0[parents]
    k = len(parents)
    A = np.zeros((k, k))
    b = np.zeros(k)
    for i in range(k):
        b[i] = np.mean(x * U[i])
        for j in range(k):
            A[i, j] = np.mean(U[j] * U[i])
    lam = sigma**2 / (s_prior**2)
    alpha_hat = np.linalg.solve(A + lam*np.eye(k), b)
    return parents, alpha_hat

def plot_alpha_errors(adj_matrix, true_alphas, initial_values, sample_sizes, target_node, sigma=0.01, s_prior=1.0):
    parents = np.where(adj_matrix[:, target_node] == 1)[0]
    if len(parents) == 0:
        print(f"Node {target_node} has no parents. Skipping alpha estimation.")
        return
    errors_mle = []
    errors_map = []
    tempdifferences=[]
    for N in sample_sizes:
        X_t0, X_t1 = simulate_graph_step(adj_matrix, true_alphas, initial_values, N=N, sigma=sigma)
        _, alpha_mle = estimate_alphas_mle(adj_matrix, X_t0, X_t1, target_node)
        _, alpha_map = estimate_alphas_map(adj_matrix, X_t0, X_t1, target_node, sigma=sigma, s_prior=s_prior)
        true_alpha_vec = np.array([true_alphas[p] for p in parents])
        errors_mle.append(np.linalg.norm(alpha_mle - true_alpha_vec))
        errors_map.append(np.linalg.norm(alpha_map - true_alpha_vec))
        tempdifferences.append(np.linalg.norm(alpha_mle - true_alpha_vec)-np.linalg.norm(alpha_map - true_alpha_vec))
    # Plot
    print(tempdifferences)
    plt.figure(figsize=(7,5))
    plt.plot(sample_sizes, errors_mle, '-o', label="MLE error")
    plt.plot(sample_sizes, errors_map, '-s', label="MAP error")
    plt.xlabel('Sample Size (N)')
    plt.ylabel('Error norm')
    plt.title(f'Node {target_node}: MLE vs MAP estimation error')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()

adj_matrix = np.array([
    [0,0,1],
    [0,0,1],
    [0,0,0]
])

##plot_graph_structure(adj_matrix)

true_alphas = np.array([1.0, 0.5, .2])   
initial_values = np.random.normal(50, 1, len(adj_matrix))
sample_sizes = [1,2,5,10, 50, 100, 500, 1000, 5000, 10000,50000,100000,500000,1000000,10000000]

plot_alpha_errors(adj_matrix, true_alphas, initial_values, sample_sizes, target_node=2, sigma=0.01, s_prior=1.0)
