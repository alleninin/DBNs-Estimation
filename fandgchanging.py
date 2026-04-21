import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

np.random.seed(18)
def simulate_nonlinear_graph_step(adj_matrix, true_alphas, initial_values, c=0.0, sigma=0.01, N=1000):
    
    n_nodes = len(adj_matrix)
    X_t0 = np.random.normal(-1, 1, size=(n_nodes, N))
    X_t1 = np.zeros((n_nodes, N))
    
    for i in range(n_nodes):
        parents = np.where(adj_matrix[:, i] == 1)[0]
        
        if len(parents) == 0:
            X_t1[i] = np.random.normal(0, 1, N)
        elif len(parents) == 2:
            alpha1, alpha2 = true_alphas[parents[0]], true_alphas[parents[1]]
            f = alpha1 * alpha2  
            g = alpha1 + alpha2  
            
            mu = f * X_t0[parents[0]] + g * X_t0[parents[1]] + c
            X_t1[i] = np.random.normal(mu, sigma)
        else:
            mu = np.zeros(N)
            for j in parents:
                mu += true_alphas[j] * X_t0[j]
            mu = mu + c
            X_t1[i] = np.random.normal(mu, sigma)
    
    return X_t0, X_t1

def estimate_nonlinear_alphas(adj_matrix, X_t0, X_t1, target_node, c=0.0):
    parents = np.where(adj_matrix[:, target_node] == 1)[0]
    
    if len(parents) != 2:
        print(f"Non-linear estimation requires exactly 2 parents, got {len(parents)}")
        return parents, np.array([])
    
    x = X_t1[target_node]
    x1 = X_t0[parents[0]]
    x2 = X_t0[parents[1]]
    
    E_xx1 = np.mean(x * x1)
    E_xx2 = np.mean(x * x2)
    E_x1x1 = np.mean(x1 * x1)
    E_x2x2 = np.mean(x2 * x2)
    E_x1x2 = np.mean(x1 * x2)
    
    A = np.array([[E_x1x1, E_x1x2],
                  [E_x1x2, E_x2x2]])
    b = np.array([E_xx1, E_xx2])
    
    try:
        fg_solution = np.linalg.solve(A, b)
        f_est, g_est = fg_solution
        
        discriminant = g_est**2 - 4*f_est
        
        if discriminant < 0:
            print(f"Complex solution - discriminant: {discriminant}")
            return parents, np.array([np.nan, np.nan])
        
        alpha1_1 = (g_est + np.sqrt(discriminant)) / 2
        alpha1_2 = (g_est - np.sqrt(discriminant)) / 2
        
        alpha2_1 = g_est - alpha1_1
        alpha2_2 = g_est - alpha1_2
        
        alpha_hat = np.array([alpha1_1, alpha2_1])
        
        return parents, alpha_hat
        
    except np.linalg.LinAlgError:
        print("Singular matrix - cannot solve system")
        return parents, np.array([np.nan, np.nan])

def plot_nonlinear_alpha_errors(adj_matrix, true_alphas, initial_values, sample_sizes, target_node):
    """
    Plot the estimation errors for different sample sizes in the non-linear case.
    """
    errors = []
    alpha_sum = np.zeros(2)  
    valid_estimates = 0
    
    for N in sample_sizes:
        X_t0, X_t1 = simulate_nonlinear_graph_step(adj_matrix, true_alphas, initial_values, N=N)
        parents, alpha_hat = estimate_nonlinear_alphas(adj_matrix, X_t0, X_t1, target_node)
        
        if len(alpha_hat) == 2 and not np.any(np.isnan(alpha_hat)):
            print(f"N={N}: α̂₁={alpha_hat[0]:.3f}, α̂₂={alpha_hat[1]:.3f}")
            alpha_sum += alpha_hat
            valid_estimates += 1
            
            true_alpha_vec = np.array([true_alphas[p] for p in parents])
            error = np.abs(alpha_hat - true_alpha_vec)
            errors.append(error)
        else:
            print(f"N={N}: Invalid solution")
            errors.append(np.array([np.nan, np.nan]))
    
    if valid_estimates > 0:
        avg_alphas = alpha_sum / valid_estimates
        print(f"Average estimates: α̂₁={avg_alphas[0]:.3f}, α̂₂={avg_alphas[1]:.3f}")
    
    errors = np.array(errors)
    
    plt.figure(figsize=(10, 6))
    
    for i, parent in enumerate(parents):
        valid_mask = ~np.isnan(errors[:, i])
        if np.any(valid_mask):
            plt.plot(np.array(sample_sizes)[valid_mask], errors[valid_mask, i], 
                    label=f'Parent {parent} → Node {target_node} (α{i+1})', marker='o')
    
    plt.xlabel('Sample Size (N)')
    plt.ylabel('|Estimated α - True α|')
    plt.title(f'Non-linear Alpha Estimation Error for Child\nf = α₁×α₂, g = α₁+α₂')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig("fandgweirdlog.svg", format="svg")
    plt.show()

if __name__ == "__main__":
    N = 3
    adj_matrix = np.array([
        [0, 0, 1],
        [0, 0, 1], 
        [0, 0, 0]
    ])
    
    true_alphas = np.array([2.0, 1.5, 0.0])  
    initial_values = np.random.normal(0, 1, N)
    
    sample_sizes = [2,4,8,16,32,64, 100, 200, 500, 1000, 2000, 5000, 10000]
    
    print("True values:")
    print(f"α₁ = {true_alphas[0]}, α₂ = {true_alphas[1]}")
    print(f"f = α₁×α₂ = {true_alphas[0] * true_alphas[1]}")
    print(f"g = α₁+α₂ = {true_alphas[0] + true_alphas[1]}")
    print()
    
    plot_nonlinear_alpha_errors(adj_matrix, true_alphas, initial_values, sample_sizes, target_node=2)
