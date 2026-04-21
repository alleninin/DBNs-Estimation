import numpy as np
import matplotlib.pyplot as plt

def generate_graph(num_nodes, adjacency=None, alpha=None, alpha_prior_std=1.0):
    if adjacency is None:
        adj = np.tril((np.random.rand(num_nodes, num_nodes) < 0.3).astype(int), -1)
    else:
        adj = np.array(adjacency)
    
    if alpha is None:
        alpha = np.random.normal(0, alpha_prior_std, size=adj.shape) * adj
    
    return adj, alpha

def generate_data(adj, alpha, n_samples, sigma=1.0):
    num_nodes = adj.shape[0]
    X = np.zeros((n_samples, num_nodes))
    order = list(range(num_nodes))  # assumes adjacency is lower triangular
    
    for t in range(n_samples):
        for i in order:
            parents = np.where(adj[i] == 1)[0]
            mean_val = X[t, parents] @ alpha[i, parents] if len(parents) > 0 else 0.0
            X[t, i] = mean_val + np.random.normal(0, sigma)
    return X

def fit_node(y, X_parents, lam=0.0):
    if X_parents.shape[1] == 0:  # no parents
        return np.array([])
    XtX = X_parents.T @ X_parents
    reg = lam * np.eye(X_parents.shape[1])
    beta = np.linalg.inv(XtX + reg) @ (X_parents.T @ y)
    return beta

def estimate_alphas(data, adj, lam=0.0):
    n, d = data.shape
    alpha_hat = np.zeros_like(adj, dtype=float)
    for i in range(d):
        parents = np.where(adj[i] == 1)[0]
        if len(parents) > 0:
            X_par = data[:, parents]
            y = data[:, i]
            beta = fit_node(y, X_par, lam)
            alpha_hat[i, parents] = beta
    print(alpha_hat)
    return alpha_hat

def run_experiment(adj, alpha_true, sigma=1.0, s_prior=1.0, sample_sizes=None, reps=1):
    if sample_sizes is None:
        sample_sizes = [10,50, 100, 200, 500, 1000,2000,5000]
    
    errors_mle = []
    errors_map = []
    
    lam = sigma**2 / (s_prior**2)  # MAP regularization weight
    
    for n in sample_sizes:
        err_mle, err_map = 0.0, 0.0
        for _ in range(reps):
            data = generate_data(adj, alpha_true, n, sigma)
            
            # MLE
            alpha_mle = estimate_alphas(data, adj, lam=0.0)
            err_mle += np.linalg.norm(alpha_mle - alpha_true)
            
            # MAP
            alpha_map = estimate_alphas(data, adj, lam=lam)
            err_map += np.linalg.norm(alpha_map - alpha_true)
        
        errors_mle.append(err_mle/reps)
        errors_map.append(err_map/reps)
    
    return sample_sizes, errors_mle, errors_map

def plot_results(sample_sizes, errors_mle, errors_map):
    plt.figure(figsize=(7,5))
    plt.plot(sample_sizes, errors_mle, '-o', label="MLE error")
    plt.plot(sample_sizes, errors_map, '-s', label="MAP error")
    plt.xlabel("Sample size (N)")
    plt.ylabel("Error norm ||α_hat - α_true||")
    plt.title("MLE vs MAP Estimation Error")
    plt.legend()
    plt.grid(True)
    plt.show()

np.random.seed(41)
##adj_matrix = np.array([
##     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
##     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
##     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
##     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
##     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
##     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
##     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
##     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
##     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
##     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

adj_matrix = np.array([
     [0, 0, 1],
     [0,0,1],
     [0,0,0]])
adj, alpha_true = generate_graph(num_nodes=5,adjacency=adj_matrix, alpha_prior_std=1.0)
print("Adjacency:\n", adj)
print("True alphas:\n", alpha_true)

sample_sizes, errors_mle, errors_map = run_experiment(adj, alpha_true, sigma=1.0, s_prior=1.0)
plot_results(sample_sizes, errors_mle, errors_map)
