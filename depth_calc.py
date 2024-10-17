import math
import numpy as np

def lambert_w(x, tol=1e-5, max_iter=100):
    """
    Compute the Lambert W function using Halley's method.

    Parameters:
    x -- The input value to the Lambert W function
    tol -- The tolerance for stopping the iteration
    max_iter -- The maximum number of iterations

    Returns:
    w -- The value of W(x) such that W(x) * e^(W(x)) = x
    """
    if x == 0:
        return 0
    if x < 0:
        return 1
#        raise ValueError("Lambert W is not defined for negative values")

    # Initial guess
    w = x

    for i in range(max_iter):
        ew = np.exp(w)
        wew = w * ew
        wew_x = wew - x
        if abs(wew_x) < tol:
            return w
        w = w - wew_x / (ew * (w + 1) - (w + 2) * wew_x / (2 * w + 2))

    raise RuntimeError("Lambert W failed to converge within the specified tolerance and maximum iterations")

def gamma_inverse(x):
    """
    (Approximate) Inverse of the gamma function.
    """
    k=1.461632 # the positive zero of the digamma function
#    assert x>=k, 'gamma(x) is strictly increasing for x >= k, k=%1.2f, x=%1.2f' % (k, x)
  #  C=math.sqrt(2*np.pi)/np.e - scipy.special.gamma(k)
    C = 0.036534
    L=np.log((x+C)/np.sqrt(2*np.pi))
    gamma_inv = 0.5+L/lambert_w(L/np.e)
    return gamma_inv

def calc_tree_depth(n_starting, max_checks):
    N_on_c = math.factorial(n_starting) / max_checks
    return int(n_starting - gamma_inverse(N_on_c))

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n = 4
    c=100000

    # Sample data
    x = list(range(n*n))
    y = [calc_tree_depth(n*n-xn,c) for xn in x]

    # Create a scatter plot
    plt.scatter(x, y, label='Data Points', color='blue', marker='o')
    plt.plot(x, y, label='Connected Lines', color='red', linestyle='--')

    # Add labels and a title
    plt.xlabel('Number of moves')
    plt.ylabel('Depth')
    plt.title(f'Maximum depth for a {n}x{n} grid with c={c}')

    plt.grid(True)

    # Display the plot
    plt.show()
