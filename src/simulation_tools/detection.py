from scipy.integrate import solve_ivp

def simulate_network(model):
    """Simulates a network
    """

    result = solve_ivp(model, [0, 1], [0])
    return result

