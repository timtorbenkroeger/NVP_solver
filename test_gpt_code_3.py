import numpy as np
from scipy.optimize import minimize

if __name__ == "__main__":


    # Parameters
    a = 250
    b = 5
    c = 8

    # Expected value of min(D(p,Z), q)
    def expected_min(p, q):
        # Since Z ~ Uniform(0,50)
        z_low, z_high = 0, 50
        def D(z):
            return a - b*p + z

        D_low = D(z_low)
        D_high = D(z_high)

        if q <= D_low:
            return q
        elif q >= D_high:
            return (D_low + D_high) / 2
        else:
            return ((q - D_low)**2 / (2*(z_high - z_low)) + q - (q - D_low) / (z_high - z_low) * (z_high - z_low))

    # Profit function
    def profit(x):
        p, q = x
        return -(p * expected_min(p, q) - c * q)

    # Bounds: price and quantity should be positive
    bounds = [(0, None), (0, None)]

    # Initial guess
    x0 = [20, 100]

    # Solve optimization
    result = minimize(profit, x0, bounds=bounds)

    # Optimal price and quantity
    p_opt, q_opt = result.x
    print(f"Optimal price: {p_opt:.2f}, Optimal quantity: {q_opt:.2f}")


