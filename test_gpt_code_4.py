import numpy as np
from scipy.optimize import minimize

if __name__ == "__main__":




    # Given parameters
    c = 8       # Unit cost
    a = 250     # Demand intercept
    b = 5       # Demand slope

    # Expected value of min(D(p, Z), q)
    def expected_min(p, q):
        # Z ~ U(0,50)
        z_low, z_high = 0, 50
        dz = z_high - z_low
        expected = 0
        num_samples = 1000
        z_samples = np.linspace(z_low, z_high, num_samples)
        d_samples = a - b * p + z_samples
        min_samples = np.minimum(d_samples, q)
        expected = np.mean(min_samples)
        return expected

    # Profit function
    def profit(x):
        p, q = x
        exp_min = expected_min(p, q)
        return -(p * exp_min - c * q)  # Negative for minimization

    # Bounds and constraints: price and quantity should be positive
    bounds = [(0.01, 100), (0.01, 500)]

    # Initial guess
    x0 = [20, 100]

    # Optimization
    result = minimize(profit, x0, bounds=bounds)

    # Optimal price and quantity
    p_opt, q_opt = result.x
    print(f"Optimal price: {p_opt:.2f}")
    print(f"Optimal quantity: {q_opt:.2f}")
    print(f"Maximum profit: {-result.fun:.2f}")



