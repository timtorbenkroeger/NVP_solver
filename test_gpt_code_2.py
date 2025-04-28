import numpy as np
from scipy.optimize import minimize

if __name__ == "__main__":


    # Parameters
    a = 250
    b = 5
    c = 8
    Z_low, Z_high = 0, 50

    # Demand function
    def demand(p, Z):
        return a - b * p + Z

    # Expected minimum of demand and quantity
    def expected_min(p, q, n_samples=10000):
        Z_samples = np.random.uniform(Z_low, Z_high, n_samples)
        D_samples = demand(p, Z_samples)
        min_samples = np.minimum(D_samples, q)
        return np.mean(min_samples)

    # Profit function to maximize
    def profit(x):
        p, q = x
        expected_sales = expected_min(p, q)
        return -(p * expected_sales - c * q)  # negative for minimization

    # Bounds for p and q
    bounds = [(0.01, 100), (0.01, 500)]  # p and q must be positive

    # Initial guess
    x0 = [20, 100]

    # Optimization
    result = minimize(profit, x0, bounds=bounds)

    optimal_p, optimal_q = result.x
    max_profit = -result.fun

    print(f"Optimal Price: {optimal_p:.2f}")
    print(f"Optimal Quantity: {optimal_q:.2f}")
    print(f"Maximum Profit: {max_profit:.2f}")

