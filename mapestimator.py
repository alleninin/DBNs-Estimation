import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns

np.random.seed(42)
x = np.sort(np.random.uniform(0, 2 * np.pi, 50))  
true_alpha = [5, 6]
sigma = 2
##xt1 = true_alpha[0] * np.cos(true_alpha[1] * x) + np.random.normal(0, sigma, size=len(x))
xt1 = true_alpha[0] * (x-true_alpha[1])**(3)-x**3.1-true_alpha[1] + np.random.normal(0, sigma, size=len(x))

 

def model(alpha, x):
    return alpha[0] * (x-alpha[1])**(3)-x**3.1-alpha[1]

def logpost(alpha, lam=1.0):
    residual = xt1 - model(alpha, x)
    likelihood = -0.5 * np.sum(residual**2) / sigma**2
    prior_term = -0.5 * lam * np.sum(np.square(alpha))  
    return -(likelihood + prior_term)  

prior = [1.0, 1.0]
result_map = minimize(logpost, prior)
alpha_map = result_map.x
print("MAP estimate:", alpha_map)

xhat = np.linspace(0, 2 * np.pi, 200)
yhat_map = model(alpha_map, xhat)

plt.plot(xhat, yhat_map, '--', label="MAP estimate")
plt.scatter(x, xt1, color='black', s=10, label='Data')
plt.title("MAP Estimate")
plt.legend()


plt.tight_layout()
plt.show()




#y_true = model(true_alpha, xhat)
#axs[0].plot(xhat, y_true, label="True", linewidth=2)


