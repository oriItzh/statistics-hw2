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
    # Get the number of possible outcomes for each variable
    n = len(X[0])  # number of outcomes for X
    m = len(Y[0])  # number of outcomes for Y
    k = len(Z[0])  # number of outcomes for Z
    
    # Joint distribution has n*m*k possible outcomes
    # But probabilities must sum to 1, so we need (n*m*k - 1) parameters
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
    # Get the number of possible outcomes for each variable
    n = len(X[0])  # number of outcomes for X
    m = len(Y[0])  # number of outcomes for Y
    k = len(Z[0])  # number of outcomes for Z
    
    # If independent, P(X,Y,Z) = P(X) * P(Y) * P(Z)
    # We need (n-1) params for P(X), (m-1) for P(Y), (k-1) for P(Z)
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
    The number of parameters that define the joint distribution of X, Y and Z if we know that X and Y are conditionally independent given Z.
    """
    # Get the number of possible outcomes for each variable
    n = len(X[0])  # number of outcomes for X
    m = len(Y[0])  # number of outcomes for Y
    k = len(Z[0])  # number of outcomes for Z
    
    # X and Y conditionally independent given Z means:
    # P(X,Y|Z) = P(X|Z) * P(Y|Z)
    # P(X,Y,Z) = P(X|Z) * P(Y|Z) * P(Z)
    # 
    # Parameters needed:
    # - P(Z): (k-1) parameters
    # - P(X|Z): For each value of Z, we need (n-1) parameters, so k*(n-1) total
    # - P(Y|Z): For each value of Z, we need (m-1) parameters, so k*(m-1) total
    # Total: (k-1) + k*(n-1) + k*(m-1) = k-1 + kn - k + km - k = kn + km - k - 1 = k(n+m-1) - 1
    return (k - 1) + k * (n - 1) + k * (m - 1)


### Question 3 ###

def my_EM(mus=np.array([4.0, 9.0, np.nan]), 
          sigmas=np.array([0.5, 0.5, 1.5]), 
          ws=np.array([np.nan, 0.25, np.nan]), 
          max_iter=1000, 
          tol=1e-6):
    """
 
    Input:          
    - mus   : a numpy array: holds the initial guess for means of the Gaussians.
              Use np.nan for parameters that need to be estimated.
    - sigmas: a numpy array: holds the initial guess for std of the Gaussians.
              Use np.nan for parameters that need to be estimated.
    - ws    : a numpy array: holds the initial guess for weights of the Gaussians.
              Use np.nan for parameters that need to be estimated.
    
    * The function should be generic and support any number of Gaussians.
      (you don't need to get this number as a parameter for the function. You can conclude it from the other parameters).
    
    Returns:
    The output of the EM algorithms (the GMM final parameters): mus, sigmas, ws.
    """
    data = pd.read_csv('GMD.csv', header=None)
    X = data.iloc[:, 1].values
    n_samples = len(X)
    k = len(mus)
    
    mus = np.array(mus, dtype=float).copy()
    sigmas = np.array(sigmas, dtype=float).copy()
    ws = np.array(ws, dtype=float).copy()
    
    # Track which parameters are fixed (not NaN)
    fixed_mus = ~np.isnan(mus)
    fixed_sigmas = ~np.isnan(sigmas)
    fixed_ws = ~np.isnan(ws)
    
    # Initialize unknown parameters with reasonable guesses
    data_mean = np.mean(X)
    data_std = np.std(X)
    
    for i in range(k):
        if np.isnan(mus[i]):
            # Initialize with spread across data range
            mus[i] = np.min(X) + (np.max(X) - np.min(X)) * (i + 1) / (k + 1)
        if np.isnan(sigmas[i]):
            sigmas[i] = data_std / k
        if np.isnan(ws[i]):
            # Calculate remaining weight
            fixed_weight_sum = np.nansum(ws[fixed_ws])
            unfixed_count = np.sum(~fixed_ws)
            ws[i] = (1 - fixed_weight_sum) / unfixed_count
    
    for iteration in range(max_iter):
        old_mus = mus.copy()
        old_sigmas = sigmas.copy()
        old_ws = ws.copy()
        
        # E-step: Calculate responsibilities
        # gamma[i, j] = P(component j | x_i)
        gamma = np.zeros((n_samples, k))
        for j in range(k):
            gamma[:, j] = ws[j] * stats.norm.pdf(X, mus[j], sigmas[j])
        
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma = gamma / gamma_sum
        
        # M-step: Update parameters
        N_j = gamma.sum(axis=0)  # effective number of points for each component
        
        for j in range(k):
            # Update mu if not fixed
            if not fixed_mus[j]:
                mus[j] = np.sum(gamma[:, j] * X) / N_j[j]
            
            # Update sigma if not fixed (with floor to prevent degenerate cases)
            if not fixed_sigmas[j]:
                sigmas[j] = max(np.sqrt(np.sum(gamma[:, j] * (X - mus[j])**2) / N_j[j]), 1e-6)
            
            # Update w if not fixed
            if not fixed_ws[j]:
                ws[j] = N_j[j] / n_samples
        
        # Renormalize weights so they sum to 1
        if not np.all(fixed_ws):
            fixed_sum = np.sum(ws[fixed_ws])
            unfixed_sum = np.sum(ws[~fixed_ws])
            if unfixed_sum > 0:
                scale = (1 - fixed_sum) / unfixed_sum
                ws[~fixed_ws] = ws[~fixed_ws] * scale
        
        mu_diff = np.max(np.abs(mus - old_mus))
        sigma_diff = np.max(np.abs(sigmas - old_sigmas))
        w_diff = np.max(np.abs(ws - old_ws))
        
        if max(mu_diff, sigma_diff, w_diff) < tol:
            break
    
    return mus, sigmas, ws

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
    # P(X < salary) using CDF of normal distribution
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
    # P(min_salary < X < max_salary) = CDF(max_salary) - CDF(min_salary)
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
    # P(X > salary) = 1 - P(X < salary) = 1 - CDF(salary)
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
    # Expected number = n_employees * P(X > salary)
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
    # T_N = sum of T_i for i=1 to N
    # where T_i is the time to collect the i-th new coupon
    # T_i follows Geometric(p_i) where p_i = (N - i + 1) / N
    # E(T_i) = 1/p_i = N / (N - i + 1)
    # E(T_N) = sum_{i=1}^{N} N/(N-i+1) = N * sum_{j=1}^{N} 1/j = N * H_N
    expected = N * sum(1/j for j in range(1, N+1))
    return expected

def CC_Variance(N=10):
    """
 
    Input:          
    - N: Number of different coupons.
    
    Returns:
    V(T_N)
    """
    # Var(T_i) = (1-p_i)/p_i^2 where p_i = (N-i+1)/N
    # Var(T_i) = (1 - (N-i+1)/N) / ((N-i+1)/N)^2
    #          = ((i-1)/N) / ((N-i+1)^2/N^2)
    #          = N*(i-1) / (N-i+1)^2
    # Since T_i are independent, Var(T_N) = sum of Var(T_i)
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
    # P(T_N > n_steps) = 1 - P(T_N <= n_steps)
    # P(T_N <= n_steps) = P(collecting all N coupons in at most n_steps draws)
    # Using inclusion-exclusion: P(all N coupons in n draws) = 
    # sum_{k=0}^{N} (-1)^k * C(N,k) * ((N-k)/N)^n
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
    The probability that S_N > n_steps (S_N is time to collect each coupon twice)
    """
    # We need to compute P(S_N > n_steps) = 1 - P(S_N <= n_steps)
    # P(S_N <= n_steps) = P(all N coupons collected at least twice in n_steps draws)
    # 
    # Using inclusion-exclusion on the event that each coupon type is collected at least twice.
    # Let A_i = event that coupon i is collected at most once (0 or 1 times)
    # P(S_N <= n) = P(none of A_i occur) = P(complement of union of A_i)
    # = 1 - P(at least one coupon collected at most once)
    # 
    # By inclusion-exclusion, P(collecting each coupon at least twice in n draws):
    # = sum over subsets S of {1,...,N}: (-1)^|S| * P(coupons in S collected at most once each)
    # 
    # P(specific k coupons collected at most once each in n draws)
    # = sum over j from 0 to min(k, n): C(k,j) * (j/N)^j * ((N-k)/N)^(n-j) * multinomial...
    # 
    # Actually, let's use a different approach with generating functions or DP.
    # 
    # Alternative: Use multinomial probability
    # P(all coupons at least twice) = sum over all valid configurations
    # This is complex. Let's use the inclusion-exclusion formula for "at least m" occurrences.
    #
    # P(each of N coupons appears at least 2 times in n draws)
    # = sum_{j=0}^{N} sum_{i=0}^{min(j,n)} (-1)^j * C(N,j) * C(j,i) * ((N-j+i)/N)^n * correction
    #
    # Actually, the cleanest formula uses:
    # P(S_N <= n) = sum_{k=0}^{N} (-1)^k * C(N,k) * sum_{j=0}^{min(k,n)} C(n,j) * (k/N)^j * ((N-k)/N)^(n-j)
    #              where the inner sum accounts for at most one occurrence of each of k coupons
    
    from scipy.special import comb
    from fractions import Fraction
    
    # More precise calculation using the multinomial approach
    # P(all N coupons appear at least twice in n draws)
    # Using inclusion-exclusion: Let B_i = "coupon i appears at most once"
    # P(all at least twice) = P(no B_i) = sum_{S} (-1)^|S| P(all i in S appear at most once)
    # 
    # P(k specific coupons each appear at most once, others unrestricted)
    # = sum_{j=0}^{min(k,n)} (ways to choose which j of the k coupons appear exactly once)
    #   * (ways to arrange j distinct items in n positions) * P(remaining n-j draws from N-k coupons)
    # = sum_{j=0}^{min(k,n)} C(k,j) * C(n,j) * j! * (1/N)^j * ((N-k)/N)^{n-j}
    # = sum_{j=0}^{min(k,n)} C(k,j) * P(n,j) * (1/N)^j * ((N-k)/N)^{n-j}
    # where P(n,j) = n!/(n-j)! is the permutation
    
    def perm(n, j):
        """Compute n!/(n-j)! = n*(n-1)*...*(n-j+1)"""
        result = 1
        for i in range(j):
            result *= (n - i)
        return result
    
    prob_complete = 0
    for k in range(N + 1):  # k = number of coupons with "at most once" restriction
        # P(these k specific coupons each appear at most once)
        inner_sum = 0
        for j in range(min(k, n_steps) + 1):  # j of these k coupons appear exactly once
            # C(k,j) ways to choose which j coupons appear exactly once
            # The other k-j don't appear at all
            # Remaining n_steps-j draws come from the other N-k coupons
            term = comb(k, j, exact=True) * perm(n_steps, j) * (Fraction(1, N) ** j) * (Fraction(N - k, N) ** (n_steps - j))
            inner_sum += term
        
        prob_complete += ((-1) ** k) * comb(N, k, exact=True) * inner_sum
    
    return float(1 - prob_complete)
