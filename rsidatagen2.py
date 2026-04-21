import numpy as np
import matplotlib.pyplot as plt

def simulate_graph_step(adj_matrix, true_alphas, initial_values, c=0.0, sigma=1.0, N=1000):
    n_nodes = len(adj_matrix)
    values_t0 = np.array(initial_values)

    X_t0 = np.tile(values_t0.reshape(-1, 1), (1, N))

    X_t1 = np.zeros((n_nodes, N))
    for i in range(n_nodes):
        parents = np.where(adj_matrix[:, i] == 1)[0]
        di = len(parents)
        if di == 0:
            X_t1[i] = np.random.normal(0, 1, N)  # Root node
        else:
            mu = np.zeros(N)
            for j in parents:
                mu += true_alphas[j] * X_t0[j]
            mu = mu / di + c
            X_t1[i] = np.random.normal(mu, sigma)
    return X_t0, X_t1

def estimate_alphas_with_constant(adj_matrix, X_t0, X_t1, target_node):
    parents = np.where(adj_matrix[:, target_node] == 1)[0]
    di = len(parents)
    if di == 0:
        return [], np.array([]), 0.0

    x = X_t1[target_node]
    U = X_t0[parents]
    k = len(parents)

    A = np.zeros((k + 1, k + 1))
    b = np.zeros(k + 1)

    b[0] = np.mean(x)
    for j in range(k):
        A[0, j + 1] = A[j + 1, 0] = np.mean(U[j]) / di
    A[0, 0] = 1  # coefficient for c

    for i in range(k):
        b[i + 1] = np.mean(x * U[i])
        A[i + 1, 0] = np.mean(U[i])  # coefficient for c
        for j in range(k):
            A[i + 1, j + 1] = np.mean(U[i] * U[j])

    solution = np.linalg.solve(A, b)
    c_hat = solution[0]
    alpha_hat = solution[1:]
    return parents, alpha_hat, c_hat

def plot_alpha_and_c_errors(adj_matrix, true_alphas, true_c, initial_values, sample_sizes, target_node):
    alpha_errors = []
    c_errors = []

    for N in sample_sizes:
        X_t0, X_t1 = simulate_graph_step(adj_matrix, true_alphas, initial_values, c=true_c, N=N)
        parents, alpha_hat, c_hat = estimate_alphas_with_constant(adj_matrix, X_t0, X_t1, target_node)

        if len(parents) == 0:
            continue

        true_alpha_vec = np.array([true_alphas[p] for p in parents])
        alpha_error = np.abs(alpha_hat - true_alpha_vec)
        c_error = np.abs(c_hat - true_c)

        alpha_errors.append(alpha_error)
        c_errors.append(c_error)

    alpha_errors = np.array(alpha_errors)
    c_errors = np.array(c_errors)

    for i, parent in enumerate(parents):
        plt.plot(sample_sizes[:len(alpha_errors)], alpha_errors[:, i], label=f'α (Parent {parent})')
    plt.xlabel('Sample Size (N)')
    plt.ylabel('|Estimated α - True α|')
    plt.title(f'Alpha Estimation Error for Node {target_node}')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(sample_sizes[:len(c_errors)], c_errors, color='black', linestyle='--', label='|Estimated c - True c|')
    plt.xlabel('Sample Size (N)')
    plt.ylabel('|Estimated c - True c|')
    plt.title(f'Constant Term Estimation Error for Node {target_node}')
    plt.legend()
    plt.grid(True)
    plt.show()
adj_matrix = np.array([
    [0, 1, 0],
    [0, 0, 0],
    [0, 1, 0]
])

true_alphas = np.array([2.5, -0.7, 3.0])
true_c = 1.2

initial_values = np.random.normal(0, 1, 3)

sample_sizes = [20, 40, 80, 160] + list(range(500, 5000, 50))

plot_alpha_and_c_errors(adj_matrix, true_alphas, true_c, initial_values, sample_sizes, target_node=1)
