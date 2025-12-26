###### Your ID ######
# ID1: 318990256
# ID2: 211882824
#####################

# imports 
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


### Question 2 ###

def q2a(X=np.array([[1, 2, 3], [0.3, 0.4, 0.3]]), 
        Y=np.array([[1, 2, 3, 4], [0.25, 0.25, 0.25, 0.25]]), 
        Z=np.array([[1, 2], [0.5, 0.5]])):
    """
 
    Input:          
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].
    
    Returns:
    The number of parameters that define the joint distribution of X, Y and Z.
    """
    n = len(X[0])
    m = len(Y[0])
    k = len(Z[0])
    return n * m * k - 1

def q2b(X=np.array([[1, 2, 3], [0.3, 0.4, 0.3]]), 
        Y=np.array([[1, 2, 3, 4], [0.25, 0.25, 0.25, 0.25]]), 
        Z=np.array([[1, 2], [0.5, 0.5]])):
    """
 
    Input:          
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].
    
    Returns:
    The number of parameters that define the joint distribution of X, Y and Z if we know that they are independent.
    """
    n = len(X[0])
    m = len(Y[0])
    k = len(Z[0])
    return (n - 1) + (m - 1) + (k - 1)

def q2c(X=np.array([[1, 2, 3], [0.3, 0.4, 0.3]]), 
        Y=np.array([[1, 2, 3, 4], [0.25, 0.25, 0.25, 0.25]]), 
        Z=np.array([[1, 2], [0.5, 0.5]])):
    """
 
    Input:          
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].
    
    Returns:
    The number of parameters that define the joint distribution of X, Y and Z if we know that they are independent.
    """
    n = len(X[0])
    m = len(Y[0])
    k = len(Z[0])
    return (k - 1) + k * (n - 1) + k * (m - 1)


### Question 3 ###

# TODO: Added 'data' parameter to match test expected signature
def my_EM(mus=np.array([4.0, 9.0, np.nan]), 
          sigmas=np.array([0.5, 0.5, 1.5]), 
          ws=np.array([np.nan, 0.25, np.nan]), 
          data=None):
    """
 
    Input:          
    - mus   : a numpy array: holds the initial guess for means of the Gaussians.
    - sigmas: a numpy array: holds the initial guess for std of the Gaussians.
    - ws    : a numpy array: holds the initial guess for weights of the Gaussians.
    
    * The function should be generic and support any number of Gaussians.
      (you don't need to get this number as a parameter for the function. You can conclude it from the other parameters).
    
    Returns:
    The output of the EM algorithms (the GMM final parameters): mus, sigmas, ws.
    """
    if data is None:
        data = pd.read_csv('GMD.csv', header=None)
        X = data.iloc[:, 1].values
    else:
        X = np.asarray(data).flatten()
    n_samples = len(X)
    k = len(mus)
    
    fixed_mus = ~np.isnan(mus)
    fixed_sigmas = ~np.isnan(sigmas)
    fixed_ws = ~np.isnan(ws)
    
    def compute_log_likelihood(X, mus, sigmas, ws):
        likelihood = np.zeros(len(X))
        for j in range(len(mus)):
            likelihood += ws[j] * stats.norm.pdf(X, mus[j], sigmas[j])
        return np.sum(np.log(likelihood + 1e-300))
    
    def run_em(mus_init, sigmas_init, ws_init, max_iter=1000, tol=1e-6):
        mus_curr = mus_init.copy()
        sigmas_curr = sigmas_init.copy()
        ws_curr = ws_init.copy()
        
        for iteration in range(max_iter):
            old_mus = mus_curr.copy()
            old_sigmas = sigmas_curr.copy()
            old_ws = ws_curr.copy()
            
            # E-step
            gamma = np.zeros((n_samples, k))
            for j in range(k):
                gamma[:, j] = ws_curr[j] * stats.norm.pdf(X, mus_curr[j], sigmas_curr[j])
            
            gamma_sum = np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
            gamma = gamma / gamma_sum
            
            # M-step
            N_j = np.maximum(gamma.sum(axis=0), 1e-10)
            
            for j in range(k):
                if not fixed_mus[j]:
                    mus_curr[j] = np.sum(gamma[:, j] * X) / N_j[j]
                
                if not fixed_sigmas[j]:
                    variance = np.sum(gamma[:, j] * (X - mus_curr[j])**2) / N_j[j]
                    sigmas_curr[j] = max(np.sqrt(variance), 1e-6)
                
                if not fixed_ws[j]:
                    ws_curr[j] = N_j[j] / n_samples
            
            if not np.all(fixed_ws):
                fixed_sum = np.sum(ws_curr[fixed_ws])
                unfixed_sum = np.sum(ws_curr[~fixed_ws])
                if unfixed_sum > 0:
                    scale = (1 - fixed_sum) / unfixed_sum
                    ws_curr[~fixed_ws] = ws_curr[~fixed_ws] * scale
            
            if max(np.max(np.abs(mus_curr - old_mus)),
                   np.max(np.abs(sigmas_curr - old_sigmas)),
                   np.max(np.abs(ws_curr - old_ws))) < tol:
                break
        
        ll = compute_log_likelihood(X, mus_curr, sigmas_curr, ws_curr)
        return mus_curr, sigmas_curr, ws_curr, ll
    
    def initialize_params(seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        mus_init = np.array(mus, dtype=float).copy()
        sigmas_init = np.array(sigmas, dtype=float).copy()
        ws_init = np.array(ws, dtype=float).copy()
        
        data_std = np.std(X)
        nan_mu_indices = np.where(~fixed_mus)[0]
        
        if len(nan_mu_indices) > 0:
            centers = list(mus_init[fixed_mus])
            for idx in nan_mu_indices:
                if len(centers) == 0:
                    mus_init[idx] = X[np.random.randint(n_samples)]
                else:
                    distances = np.array([min((x - c)**2 for c in centers) for x in X])
                    distances = distances / (distances.sum() + 1e-10)
                    chosen_idx = np.random.choice(n_samples, p=distances)
                    mus_init[idx] = X[chosen_idx]
                centers.append(mus_init[idx])
        
        for i in range(k):
            if np.isnan(sigmas_init[i]):
                distances = np.abs(X - mus_init[i])
                nearby_mask = distances < data_std
                if np.sum(nearby_mask) > 10:
                    sigmas_init[i] = np.std(X[nearby_mask])
                else:
                    sigmas_init[i] = data_std / k
                sigmas_init[i] = max(sigmas_init[i], 0.1)
        
        for i in range(k):
            if np.isnan(ws_init[i]):
                fixed_weight_sum = np.nansum(ws_init[fixed_ws])
                unfixed_count = np.sum(~fixed_ws)
                ws_init[i] = (1 - fixed_weight_sum) / unfixed_count
        
        return mus_init, sigmas_init, ws_init
    
    n_restarts = 10
    best_ll = -np.inf
    best_result = None
    
    for restart in range(n_restarts):
        seed = restart * 42 if restart > 0 else None
        mus_init, sigmas_init, ws_init = initialize_params(seed)
        mus_res, sigmas_res, ws_res, ll = run_em(mus_init, sigmas_init, ws_init)
        
        if ll > best_ll:
            best_ll = ll
            best_result = (mus_res.copy(), sigmas_res.copy(), ws_res.copy())
    
    return best_result

# TODO: Added 'n_samples' parameter to match test expected signature
def q3d(mus=np.array([4.0, 9.0, 15.0]), 
        sigmas=np.array([0.5, 0.5, 1.5]), 
        ws=np.array([0.125, 0.25, 0.625]), 
        n_samples=1000):
    """
 
    Input:          
    - mus   : a numpy array: holds the means of the gaussians.
    - sigmas: a numpy array: holds the stds of the gaussians.
    - ws    : a numpy array: holds the weights of the gaussians.
    
    * The function should be generic and support any number of Gaussians.
      (you don't need to get this number as a parameter for the function. You can conclude it from the other parameters).
    
    Returns:
    The generated data.
    """
    mus = np.array(mus)
    sigmas = np.array(sigmas)
    ws = np.array(ws)
    
    k = len(mus)
    component_indices = np.random.choice(k, size=n_samples, p=ws)
    samples = np.random.normal(mus[component_indices], sigmas[component_indices])
    
    return samples


### Question 4 ###

def q4a(mu=75000, sigma=37500, salary=50000):
    """
 
    Input:          
    - mu   : The mean of the annual salaries of employees in a large Randomistan company.
    - sigma: The std of the annual salaries of employees in a large Randomistan company.
    The annual salary of employees in a large Randomistan company are approximately normally distributed.
    
    Returns:
    The percent of people earn less than 'salary'.
    """
    percent = stats.norm.cdf(salary, mu, sigma) * 100
    return percent

def q4b(mu=75000, sigma=37500, min_salary=45000, max_salary=65000):
    """
 
    Input:          
    - mu   : The mean of the annual salaries of employees in a large Randomistan company.
    - sigma: The std of the annual salaries of employees in a large Randomistan company.
    The annual salary of employees in a large Randomistan company are approximately normally distributed.
    
    Returns:
    The percent of people earn between 'min_salary' and 'max_salary'.
    """
    percent = (stats.norm.cdf(max_salary, mu, sigma) - stats.norm.cdf(min_salary, mu, sigma)) * 100
    return percent

def q4c(mu=75000, sigma=37500, salary=85000):
    """
 
    Input:          
    - mu   : The mean of the annual salaries of employees in a large Randomistan company.
    - sigma: The std of the annual salaries of employees in a large Randomistan company.
    The annual salary of employees in a large Randomistan company are approximately normally distributed.
    
    Returns:
    The percent of people earn more than 'salary'.
    """
    percent = (1 - stats.norm.cdf(salary, mu, sigma)) * 100
    return percent

def q4d(mu=75000, sigma=37500, salary=140000, n_employees=1000):
    """
 
    Input:          
    - mu         : The mean of the annual salaries of employees in a large Randomistan company.
    - sigma      : The std of the annual salaries of employees in a large Randomistan company.
    - n_employees: The number of employees in the company
    The annual salary of employees in a large Randomistan company are approximately normally distributed.
    
    Returns:
    The number of employees in the company that you expect to earn more than 'salary'.
    """
    prob = 1 - stats.norm.cdf(salary, mu, sigma)
    expected = n_employees * prob
    return expected


### Question 5 ###

def CC_Expected(N=10):
    """
 
    Input:          
    - N: Number of different coupons.
    
    Returns:
    E(T_N)
    """
    expected = N * sum(1/j for j in range(1, N+1))
    return expected

def CC_Variance(N=10):
    """
 
    Input:          
    - N: Number of different coupons.
    
    Returns:
    V(T_N)
    """
    variance = 0
    for i in range(1, N+1):
        p_i = (N - i + 1) / N
        var_i = (1 - p_i) / (p_i ** 2)
        variance += var_i
    return variance

def CC_T_Steps(N=10, n_steps=30):
    """
 
    Input:          
    - N: Number of different coupons.
    
    Returns:
    The probability that T_N > n_steps
    """
    from scipy.special import comb
    
    prob_complete = 0
    for k in range(N + 1):
        prob_complete += ((-1) ** k) * comb(N, k, exact=True) * ((N - k) / N) ** n_steps
    
    return 1 - prob_complete

def CC_S_Steps(N=10, n_steps=30):
    """
 
    Input:          
    - N: Number of different coupons.
    
    Returns:
    The probability that S_N > n_steps
    """
    from scipy.special import comb
    from fractions import Fraction
    
    def perm(n, j):
        result = 1
        for i in range(j):
            result *= (n - i)
        return result
    
    prob_complete = 0
    for k in range(N + 1):
        inner_sum = 0
        for j in range(min(k, n_steps) + 1):
            term = comb(k, j, exact=True) * perm(n_steps, j) * (Fraction(1, N) ** j) * (Fraction(N - k, N) ** (n_steps - j))
            inner_sum += term
        
        prob_complete += ((-1) ** k) * comb(N, k, exact=True) * inner_sum
    
    return float(1 - prob_complete)
