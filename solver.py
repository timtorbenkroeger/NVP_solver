import numpy as np

from scipy.stats import norm
import math

def optimal_newsvendor_quantity(sales_price, unit_order_cost, mean, variance):
    # Critical ratio
    critical_ratio = (sales_price - unit_order_cost) / (sales_price )

    # Standard deviation from variance
    std_dev = math.sqrt(variance)

    # Optimal order quantity
    optimal_quantity = norm.ppf(critical_ratio, loc=mean, scale=std_dev)

    return optimal_quantity




def find_optimal_price_quantity(a: float, b: float, c: float, z_lower: float, z_upper: float):
    """
    Computes the profit-maximizing price (p*) and quantity (q*) for a
    price-setting newsvendor.

    Assumes demand D(p) = a - b*p + Z, where Z ~ Uniform(z_lower, z_upper).

    Args:
        a: Demand intercept.
        b: Price sensitivity coefficient (must be > 0).
        c: Unit order cost (must be >= 0).
        z_lower: Lower bound of the uniform distribution for Z.
        z_upper: Upper bound of the uniform distribution for Z (must be >= z_lower).

    Returns:
        A tuple (p_star, q_star) representing the optimal price and quantity.
        Returns (None, None) if no valid solution is found (e.g., b <= 0,
        z_upper < z_lower, or no real root p > c exists).

    Raises:
        ValueError: If input parameters are invalid (b<=0, c<0, z_upper<z_lower).
    """
    # --- Input Validation ---
    if b <= 0:
        raise ValueError("Price sensitivity 'b' must be positive.")
    if c < 0:
        raise ValueError("Unit cost 'c' must be non-negative.")
    if z_upper < z_lower:
        raise ValueError("z_upper must be greater than or equal to z_lower.")

    # --- Calculate Intermediate Values ---
    # Width of the uniform distribution
    W = z_upper - z_lower
    # Expected value of Z
    E_Z = (z_lower + z_upper) / 2.0

    # --- Define Coefficients of the Cubic Equation for p* ---
    # The equation derived from dI/dp = 0 is:
    # 2*b*p^3 - (a + E[Z] + b*c)*p^2 + (W * c^2 / 2) = 0
    coeff3 = 2.0 * b
    coeff2 = -(a + E_Z + b * c)
    coeff1 = 0.0
    coeff0 = (W * c**2) / 2.0

    coefficients = [coeff3, coeff2, coeff1, coeff0]

    # --- Solve the Cubic Equation for p ---
    try:
        roots = np.roots(coefficients)
    except np.linalg.LinAlgError:
        print("Error solving polynomial roots.")
        return None, None

    # --- Filter Roots to Find Valid p* ---
    valid_p_star = None
    max_profit = -np.inf # Initialize with negative infinity

    # Define the expected profit function I(p) for validation if multiple roots
    # I(p) = (p-c) * [ a + E_Z - b*p - (W*c)/(2*p) ]
    def expected_profit(p_val):
        if p_val <= 0: return -np.inf # Avoid division by zero / negative prices
        return (p_val - c) * (a + E_Z - b * p_val - (W * c) / (2.0 * p_val))

    potential_p_values = []
    for p_val in roots:
        # Check if the root is real and economically meaningful (p > c)
        if np.isreal(p_val) and p_val.real > c:
            potential_p_values.append(p_val.real)

    if not potential_p_values:
        print(f"No valid real root found for p > c (c={c}). Roots found: {roots}")
        return None, None

    # If there's more than one valid root p > c, choose the one maximizing profit
    # (Often only one exists, but checking is safer)
    if len(potential_p_values) > 1:
        print(f"Warning: Multiple potential p* values found ({potential_p_values}). Selecting the one maximizing profit.")
        best_p = -1
        max_prof = -np.inf
        for p_test in potential_p_values:
            prof = expected_profit(p_test)
            if prof > max_prof:
                max_prof = prof
                best_p = p_test
        valid_p_star = best_p
    else:
        valid_p_star = potential_p_values[0]


    if valid_p_star is None:
        print(f"Could not determine a unique valid p* from roots: {roots}")
        return None, None

    p_star = valid_p_star

    # --- Calculate Optimal Quantity q* ---
    # q*(p) = (a + z_upper) - b*p - (W*c)/p
    try:
        q_star = (a + z_upper) - b * p_star - (W * c) / p_star
    except ZeroDivisionError:
        print("Error: Division by zero encountered when calculating q* (p* is likely zero).")
        return None, None


    # --- Optional Check: Ensure q* falls within reasonable demand range ---
    # D_min(p*) = a - b*p* + z_lower
    # D_max(p*) = a - b*p* + z_upper
    # If q* is outside [D_min, D_max] it might indicate an issue,
    # but the derived formula should inherently handle this if p* is correct.

    return p_star, q_star
