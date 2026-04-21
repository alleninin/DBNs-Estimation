import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def simulate_graph_step(adj_matrix, true_alphas, initial_values, c=0.0, sigma=0.01, N=1000):
    n_nodes = len(adj_matrix)
    values_t0 = np.array(initial_values)
    X_t0 = np.random.normal(50, 1, size=(n_nodes, N))
    X_t1 = np.zeros((n_nodes, N))
    for i in range(n_nodes):
        parents = np.where(adj_matrix[:, i] == 1)[0]
        di = len(parents)
        if di == 0:
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
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=700, node_color='lightblue', edge_color='gray')
    plt.title("Graph Structure (DAG)")
    plt.show()


def estimate_alphas(adj_matrix, X_t0, X_t1, target_node, c=0.0):
    parents = np.where(adj_matrix[:, target_node] == 1)[0]
    if len(parents) == 0:
        return [], np.array([])
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

def plot_alpha_errors(adj_matrix, true_alphas, initial_values, sample_sizes, target_node):
    parents = np.where(adj_matrix[:, target_node] == 1)[0]
    if len(parents) == 0:
        print(f"Node {target_node} has no parents. Skipping alpha estimation.")
        return
    errors = []
    alpha_sum = np.zeros(len(parents))
    for N in sample_sizes:
        X_t0, X_t1 = simulate_graph_step(adj_matrix, true_alphas, initial_values, N=N)
        parents, alpha_hat = estimate_alphas(adj_matrix, X_t0, X_t1, target_node)
        if len(alpha_hat) == 0:
            print(f"  Skipping N={N} for node {target_node} (no parents)")
            continue
        print(f"N={N}: {alpha_hat}")
        alpha_sum += alpha_hat
        true_alpha_vec = np.array([true_alphas[p] for p in parents])
        error = np.abs(alpha_hat - true_alpha_vec)
        errors.append(error)
    if len(errors) == 0:
        print(f"No valid alpha estimates collected for node {target_node}.")
        return
    avg_alphas = alpha_sum / len(errors)
    print(f"Average estimates: {avg_alphas}")
    errors = np.array(errors)
    for i, parent in enumerate(parents):
        plt.plot(sample_sizes[:len(errors)], errors[:, i], label=f'Parent {parent} → Node {target_node}')
    plt.xlabel('Sample Size (N)')
    plt.ylabel('|Estimated α - True α|')
    plt.title(f'Alpha Estimation Error for Node {target_node}')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

##test_simple_graph()
##
##print("\n" + "="*50)
##print("Original example from your code:")
##print("="*50)

N = 8
adj_matrix = np.array([
    [0,0,1,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,1,1,0,0],
    [0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0]
])

plot_graph_structure(adj_matrix)


##N = 4
##adj_matrix = np.array([
##    [0,0,1,0],
##    [0,0,1,0],
##    [0,0,0,0],
##    [0,0,1,0]
##])

true_alphas = np.array([1.0, 2.0, 0.5, 1.5, 0.8, 1.2, 2.2, 3.3])
initial_values = np.random.normal(50, 1, len(adj_matrix))
sample_sizes = [1, 10, 100, 1000, 10000, 100000]
sample_sizes += list(range(0, 100000, 1000))
num_trials = 100

for node in range(len(adj_matrix)):
    print(f"\nEstimating alphas for node {node}")
    plot_alpha_errors(adj_matrix, true_alphas, initial_values, sample_sizes, target_node=node)

print(true_alphas)
