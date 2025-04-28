import numpy as np

if __name__ == "__main__":
    # Parameters
    a = 250
    b = 5
    c = 8
    z_low, z_high = 0, 50

    # Discretize price and quantity
    price_grid = np.linspace(10, 80, 100)
    quantity_grid = np.linspace(10, 400, 100)

    # Function to compute expected sales E[min(D(p,Z),q)]
    def expected_sales(p, q, num_samples=10000):
        Z_samples = np.random.uniform(z_low, z_high, num_samples)
        D_samples = a - b*p + Z_samples
        sales = np.minimum(D_samples, q)
        return np.mean(sales)

    # Search for the best (p,q)
    best_profit = -np.inf
    best_pq = (None, None)

    for p in price_grid:
        for q in quantity_grid:
            exp_sales = expected_sales(p, q)
            profit = p * exp_sales - c * q
            if profit > best_profit:
                best_profit = profit
                best_pq = (p, q)
    print("test")
    print(f"Optimal price: {best_pq[0]:.2f}")
    print(f"Optimal quantity: {best_pq[1]:.2f}")
    print(f"Maximum expected profit: {best_profit:.2f}")
