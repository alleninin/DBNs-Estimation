import numpy as np
import matplotlib.pyplot as plt

def simulate_graph_step(adj_matrix, true_alphas, initial_values, c=0.0, sigma=0.01, N=1000):
    n_nodes = len(adj_matrix)
    values_t0 = np.array(initial_values)
    X_t0 = np.random.normal(-1, 1, size=(n_nodes, N))
##    print(X_t0)
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
    """
    Plot the estimation errors for different sample sizes.
    """
    errors = []
    alpha_sum = np.zeros(len(np.where(adj_matrix[:, target_node] == 1)[0]))
    
    for N in sample_sizes:
        X_t0, X_t1 = simulate_graph_step(adj_matrix, true_alphas, initial_values, N=N)
        parents, alpha_hat = estimate_alphas(adj_matrix, X_t0, X_t1, target_node)
##        parents, alpha_hat = estimate_alphas(adj_matrix, np.array([[0,2,1],[1,1,4],[0,0,0]]), np.array([[0,2,1],[1,1,4],[0.5,1.5,2.8]]), target_node)

        print(f"N={N}: {alpha_hat}")
        alpha_sum += alpha_hat
        
        true_alpha_vec = np.array([true_alphas[p] for p in parents])
        error = np.abs(alpha_hat - true_alpha_vec)
        errors.append(error)
    
    avg_alphas = alpha_sum / len(sample_sizes)
    print(f"Average estimates: {avg_alphas}")
    
    errors = np.array(errors)
    for i, parent in enumerate(parents):
        plt.plot(sample_sizes, errors[:, i], label=f'Parent {parent} → Node {target_node}')
    
    plt.xlabel('Sample Size (N)')
    plt.ylabel('|Estimated α - True α|')
    plt.title(f'Alpha Estimation Error for Node {target_node}')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig("3nodeestimate.svg", format="svg")

    plt.show()


##N = 4
##adj_matrix = np.array([
##    [0,0,1,0],
##    [0,0,1,0],
##    [0,0,0,0],
##    [0,0,1,0]
##])
N = 4
adj_matrix = np.array([
    [0,0,1,0],
    [0,0,1,0],
    [0,0,0,0],
    [0,0,1,0]
])

true_alphas = np.array([2.5, 0, 3.0, 4.0])
initial_values = np.random.normal(50, 1, N)
sample_sizes = [1,3,10,30,100,300,1000,3000,10000]
num_trials = 100

plot_alpha_errors(adj_matrix, true_alphas, initial_values, sample_sizes, target_node=2)
##print(true_alphas)
