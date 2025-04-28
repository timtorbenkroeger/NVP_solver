import solver

from scipy.stats import norm

if __name__ == "__main__":
    # print(solver.find_optimal_price_quantity(250,5,8,0,50))
    #print(norm.ppf(0.4, loc=50, scale=5))
    #print(norm.ppf(0.4))
    #print(solver.optimal_newsvendor_quantity(10,8,50,50))
    print(solver.find_optimal_price_quantity(250,5,8,0,50))

