import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import networkx as nx

np.random.seed(42)
meanalphas=2
L=100
def simulate_graph(adj_matrix, true_alphas, nonlinear_func, sigma=0.01, N=1000):
    n_nodes = len(adj_matrix)
    X_t0 = np.random.normal(0, 1, size=(n_nodes, N))
    X_t1 = np.zeros((n_nodes, N))

    for i in range(n_nodes):
        parents = np.where(adj_matrix[:, i] == 1)[0]
        if len(parents) == 0:
            X_t1[i] = np.random.normal(0, 1, N)  # root noise
        else:
            eq_str = f"Node {i}: X_t1[{i}] = "
            terms = []
            for j in parents:
                alpha_val = true_alphas[(j, i)]
                terms.append(f"{alpha_val:.3f} * (X_t0[{j}]^2)")
            eq_str += " + ".join(terms) + " + noise"
            print(eq_str)
            inputs = [true_alphas[(j, i)] * (X_t0[j]) for j in parents]  # a * x^2
            mu = nonlinear_func(*inputs)
            X_t1[i] = np.random.normal(mu, sigma)
    return X_t0, X_t1


def plot_graph_structure(adj_matrix):
    G = nx.DiGraph()
    n = len(adj_matrix)
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6, 5))
    nx.draw(G, pos, with_labels=True, arrows=True,
            node_size=600, node_color='lightblue', edge_color='gray')
    plt.title("Graph Structure")
    plt.show()


def estimate_alphas_mle(adj_matrix, X_t0, X_t1, nonlinear_func, target_edges, sigma2=0.01**2):
    parent_dict = {i: np.where(adj_matrix[:, i] == 1)[0] for i in range(adj_matrix.shape[0])}
    edge_index = {edge: k for k, edge in enumerate(target_edges)}

    def loss(alpha_vec):
        data_term = 0.0
        for i in range(adj_matrix.shape[0]):
            parents = parent_dict[i]
            if len(parents) == 0:
                continue
            coeffs = alpha_vec[[edge_index[(p, i)] for p in parents]]
            mu = np.dot(coeffs, X_t0[parents])
            data_term += np.sum((X_t1[i] - mu) ** 2) / (2 * sigma2)
        return data_term

    init_alpha = np.ones(len(target_edges))
    res = minimize(loss, init_alpha, method="BFGS")
    return res.x


def estimate_alphas_map(adj_matrix, X_t0, X_t1, nonlinear_func, target_edges,
                        sigma2=0.1**2, tau2=0.001**2, mu_prior_val=meanalphas):
    mu_prior = np.full(len(target_edges), mu_prior_val)
    parent_dict = {i: np.where(adj_matrix[:, i] == 1)[0] for i in range(adj_matrix.shape[0])}
    edge_index = {edge: k for k, edge in enumerate(target_edges)}

    def loss(alpha_vec):
        data_term = 0.0
        for i in range(adj_matrix.shape[0]):
            parents = parent_dict[i]
            if len(parents) == 0:
                continue
            coeffs = alpha_vec[[edge_index[(p, i)] for p in parents]]
            mu = np.dot(coeffs, X_t0[parents])
            data_term += np.sum((X_t1[i] - mu) ** 2) / (2 * sigma2)
        prior_term = np.sum((alpha_vec - mu_prior) ** 2) / (2 * tau2)
        return data_term + prior_term

    init_alpha = np.ones(len(target_edges)) * mu_prior_val
    res = minimize(loss, init_alpha, method="BFGS")
    return res.x


def estimate_alphas_map_uniform(adj_matrix, X_t0, X_t1, nonlinear_func, target_edges,
                                sigma2=0.01**2, center=meanalphas, width=0.5):
    bounds = [(center - width, center + width) for _ in target_edges]
    parent_dict = {i: np.where(adj_matrix[:, i] == 1)[0] for i in range(adj_matrix.shape[0])}
    edge_index = {edge: k for k, edge in enumerate(target_edges)}

    def loss(alpha_vec):
        data_term = 0.0
        for i in range(adj_matrix.shape[0]):
            parents = parent_dict[i]
            if len(parents) == 0:
                continue
            coeffs = alpha_vec[[edge_index[(p, i)] for p in parents]]
            mu = np.dot(coeffs, X_t0[parents])
            data_term += np.sum((X_t1[i] - mu) ** 2) / (2 * sigma2)
        return data_term

    init_alpha = np.ones(len(target_edges)) * center
    res = minimize(loss, init_alpha, method="L-BFGS-B", bounds=bounds)
    return res.x

def generate_dag_adjacency(n=100, num_roots=1, edge_prob=0.1):
    
    adj_matrix = np.zeros((n, n), dtype=int)
    
    roots = list(range(num_roots))      
    for i in range(num_roots, n):
        possible_parents = list(range(i))
        parent = np.random.choice(possible_parents)
        adj_matrix[parent, i] = 1
    
    for i in range(n):
        for j in range(i+1, n):
            if adj_matrix[i, j] == 0 and np.random.random() < edge_prob:
                adj_matrix[i, j] = 1
    
    return adj_matrix



def run_experiment(adj_matrix, nonlinear_func, sample_sizes,
                   sigma2=0.01**2, tau2=0.1**2, force_alphas_two=False):
    edges = [(i, j) for i in range(len(adj_matrix)) for j in range(len(adj_matrix)) if adj_matrix[i, j] == 1]

    if force_alphas_two:
        true_alphas = {edge: meanalphas for edge in edges}
    else:
        true_alphas = {edge: np.random.normal(meanalphas, 0.01) for edge in edges}

    errors_mle, errors_map,anothaone = [], [],[]

    for N in sample_sizes:
        X_t0, X_t1 = simulate_graph(adj_matrix, true_alphas, nonlinear_func, sigma=np.sqrt(sigma2), N=N)

        alpha_hat_mle = estimate_alphas_mle(adj_matrix, X_t0, X_t1, nonlinear_func, edges, sigma2=sigma2)
        alpha_hat_map = estimate_alphas_map(adj_matrix, X_t0, X_t1, nonlinear_func, edges,
                                            sigma2=sigma2, tau2=tau2, mu_prior_val=meanalphas)

        true_alpha_vec = np.array([true_alphas[edge] for edge in edges])
        error_mle = np.linalg.norm(alpha_hat_mle - true_alpha_vec)/(true_alpha_vec.size)
        #error_map = np.linalg.norm(alpha_hat_map - true_alpha_vec)/(true_alpha_vec.size)
        alpha_hat_map_uniform = estimate_alphas_map_uniform(adj_matrix, X_t0, X_t1,
                                                    nonlinear_func, edges,
                                                    sigma2=sigma2, center=meanalphas, width=0.1)
        error_map_uniform = np.linalg.norm(alpha_hat_map_uniform - true_alpha_vec)/(true_alpha_vec.size)

        errors_mle.append(error_mle)
        #errors_map.append(error_map)
        anothaone.append(error_map_uniform)
        print(f"N={N}, MLE Error={error_mle:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(sample_sizes, errors_mle, "o-", label="MLE")
    #plt.plot(sample_sizes, errors_map, "s-", label="Gaussian Priors")
    plt.plot(sample_sizes, anothaone, "s-", label="Uniform Distribution")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Sample size (N)")
    plt.ylabel("Alpha Error Norm/Num alphas")
    plt.title("MLE vs MAP Estimation Error for graph with "+str(L)+" nodes, mean at "+str(meanalphas))
    plt.grid(True)
    plt.legend()
    plt.savefig("100nodes.svg", format="svg")
    
    plt.show()
if __name__ == "__main__":
    '''adj_matrix = np.array([
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])'''

    #high connected, high roots
##    adj_matrix = np.array([[0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
##     [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
##     [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
##     [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
##     [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
##     [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
##     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
##     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
##     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
##     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
##    adj_matrix = np.array([
##     [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
##     [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
##     [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
##     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
##     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
##     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
##     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
##     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
##     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
##     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    adj_matrix = generate_dag_adjacency(L, num_roots=3, edge_prob=0.05)

    #plot_graph_structure(adj_matrix)

    edges = [(i, j) for i in range(len(adj_matrix)) for j in range(len(adj_matrix)) if adj_matrix[i, j] == 1]
    true_alphas = {edge: np.random.normal(meanalphas, 0.01) for edge in edges}

    def nonlinear_func(*args):
        return (np.sum(args, axis=0))

    sample_sizes = [2, 4, 10, 16, 32]
    #sample_sizes = [2, 4, 10, 16,32]

    run_experiment(adj_matrix, nonlinear_func, sample_sizes,
                   sigma2=0.01**2, tau2=0.001**2, force_alphas_two=True)


