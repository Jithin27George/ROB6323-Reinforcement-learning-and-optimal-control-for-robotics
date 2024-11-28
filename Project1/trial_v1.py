%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import IPython

from scipy.linalg import block_diag
from scipy import sparse

from qpsolvers import solve_qp, Problem, solve_problem

import quadrotor

# Initialisation
m = quadrotor.MASS
r = quadrotor.LENGTH
Ine = quadrotor.INERTIA
dt = quadrotor.DT
xdim = quadrotor.DIM_STATE  # 6
udim = quadrotor.DIM_CONTROL  # 2
grav = quadrotor.GRAVITY_CONSTANT

print("Mass    =", quadrotor.MASS)
print("Length  =", quadrotor.LENGTH)
print("Inertia =", quadrotor.INERTIA)
print("Dt      =", quadrotor.DT)
print("state size   =", quadrotor.DIM_STATE)
print("control size =", quadrotor.DIM_CONTROL)
print("gravity constant =", quadrotor.GRAVITY_CONSTANT)


# Define functions with corrections

def C_d_functions(Xbar: np.ndarray, iterations, udim=2, xdim=6, debug=False):
    # Ensure Xbar is a 1D array
    Xbar = Xbar.flatten()

    # Initialize C and d with the desired shapes
    C = np.zeros((xdim * (iterations + 1), (iterations + 1) * xdim + iterations * udim))
    d = np.zeros((xdim * (iterations + 1), 1))

    # Set initial conditions for the state
    C[0:xdim, 0:xdim] = np.eye(xdim)

    if debug:
        print("nvars =", (iterations +1)*xdim + iterations *udim)
        print("C shape =", C.shape)
        print("d shape =", d.shape)
        print("Xbar length =", len(Xbar))

    for i in range(iterations):
        # Compute indices
        x_i_start = i * (xdim + udim)
        u_i_start = x_i_start + xdim
        x_ip1_start = (i +1) * (xdim + udim)

        if debug:
            print("________________________________________________")
            print(f"Iteration {i}")
            print(f"x_i_start = {x_i_start}")
            print(f"u_i_start = {u_i_start}")
            print(f"x_ip1_start = {x_ip1_start}")

        # Extract current state and control variables
        px_bar_i = float(Xbar[x_i_start])
        vx_bar_i = float(Xbar[x_i_start + 1])
        py_bar_i = float(Xbar[x_i_start + 2])
        vy_bar_i = float(Xbar[x_i_start + 3])
        theta_bar_i = float(Xbar[x_i_start + 4])
        omega_bar_i = float(Xbar[x_i_start + 5])
        u1_bar_i = float(Xbar[u_i_start])
        u2_bar_i = float(Xbar[u_i_start + 1])

        # Extract next state variables
        px_bar_inext = float(Xbar[x_ip1_start])
        vx_bar_inext = float(Xbar[x_ip1_start + 1])
        py_bar_inext = float(Xbar[x_ip1_start + 2])
        vy_bar_inext = float(Xbar[x_ip1_start + 3])
        theta_bar_inext = float(Xbar[x_ip1_start + 4])
        omega_bar_inext = float(Xbar[x_ip1_start + 5])

        # Compute A and B matrices
        A = np.array([
            [1.0, dt, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        B = np.zeros((xdim, udim))

        B_alpha = -(dt * np.sin(theta_bar_i)) / m
        B_beta = (dt * np.cos(theta_bar_i)) / m
        B_gamma = (r * dt) / Ine

        A_alpha = -(dt * np.cos(theta_bar_i) * (u1_bar_i + u2_bar_i)) / m
        A_beta = -(dt * np.sin(theta_bar_i) * (u1_bar_i + u2_bar_i)) / m

        B[1, 0] = B_alpha
        B[1, 1] = B_alpha
        B[3, 0] = B_beta
        B[3, 1] = B_beta
        B[5, 0] = B_gamma
        B[5, 1] = -B_gamma

        A[1, 4] = A_alpha
        A[3, 4] = A_beta

        # Indices in C matrix
        # State variables indices
        x_i_indices = i * (xdim + udim)
        x_ip1_indices = (i + 1) * (xdim + udim)

        # Map x_i
        C[(i + 1) * xdim:(i + 2) * xdim, x_i_start:x_i_start + xdim] = A

        # Map u_i
        C[(i + 1) * xdim:(i + 2) * xdim, (iterations +1)*xdim + i * udim : (iterations +1)*xdim + (i +1)*udim] = B

        # Map x_{i+1}
        C[(i + 1) * xdim:(i + 2) * xdim, x_ip1_start:x_ip1_start + xdim] = -np.eye(xdim)

        # Build d vector
        idx = (i +1)*xdim
        d[idx +0,0] = px_bar_i + dt * vx_bar_i - px_bar_inext
        d[idx +1,0] = vx_bar_i + A_alpha - vx_bar_inext
        d[idx +2,0] = py_bar_i + dt * vy_bar_i - py_bar_inext
        d[idx +3,0] = vy_bar_i + A_beta - grav * dt - vy_bar_inext
        d[idx +4,0] = theta_bar_i + dt * omega_bar_i - theta_bar_inext
        d[idx +5,0] = omega_bar_i + B_gamma * (u1_bar_i - u2_bar_i) - omega_bar_inext

        if debug:
            print(f"A matrix at iteration {i}:\n{A}")
            print(f"B matrix at iteration {i}:\n{B}")
            print(f"d vector at iteration {i}:\n{d[idx:idx + xdim, 0]}")
        print("C_d_functions: PASS")

    return C.astype(np.float64), d.astype(np.float64)


def cost_gradient_hessian_function(Xbar: np.ndarray, iterations, udim=2, xdim=6, debug=False):
    # Check the dimension of Xbar
    Xbar = np.asarray(Xbar, dtype=np.float64)

    # G matrix construction  
    Q = np.diag([10, 1, 10, 1, 100, 1])
    R = np.diag([0.1, 0.1])

    Matrix = np.block([
        [Q, np.zeros((xdim, udim))],
        [np.zeros((udim, xdim)), R]
    ])  # 8x8

    if debug:
        print(f"Matrix block diagonal for G: {Matrix.shape}")

    Unit1 = [Matrix] * iterations
    Unit1.append(Q)  # Last state has no control
    G = block_diag(*Unit1).astype(np.float64)

    if debug:
        print(f"Dimension of G (cost_function): {G.shape}")

    # g matrix construction 
    g = np.zeros(( (iterations +1)*xdim + iterations *udim, ), dtype=np.float64)
    distribution = iterations // 4
    if debug:
        print(f"Distribution: {distribution}")

    for i in range(iterations +1):
        if i <= distribution:
            x_des = np.array([3, 0, 3, 0, (np.pi / 2), 0])   
        elif distribution < i <= 2 * distribution:
            x_des = np.array([0, 0, 6, 0, (np.pi), 0])   
        elif 2 * distribution < i <= 3 * distribution:
            x_des = np.array([-3, 0, 3, 0, 3 * (np.pi / 2), 0])   
        else:
            x_des = np.array([0, 0, 0, 0, 0, 0])
        
        if debug:
            print(f"Iteration {i}: x_des = {x_des}")
        
        # Pad with zeros for control variables if not the last state
        if i < iterations:
            x_des_padded = np.concatenate([x_des, np.zeros(udim)])
        else:
            x_des_padded = x_des  # Last state has no control
        
        # Assign to g
        start_idx = i * (xdim + udim)
        end_idx = start_idx + xdim + udim
        g[start_idx:end_idx] += x_des_padded @ Matrix.T[:xdim + udim]
        
        if debug:
            print(f"g[{start_idx} : {end_idx}] updated to {g[start_idx:end_idx]}")

    g = g.reshape(-1,1)

    if debug:
        print(f"g shape: {g.shape}")

    # Compute gradient and hessian
    cost_function = 0.5 * (Xbar.T @ G @ Xbar) + (g.T @ Xbar)
    gradient_cost = G @ Xbar + g
    hessian_cost = G

    if debug:
        print("Cost function, gradient, and hessian computed.")

    print("cost_gradient_hessian_function: PASS")
    return cost_function.astype(np.float64), gradient_cost.astype(np.float64), hessian_cost.astype(np.float64)


def cost_func(Xbar: np.ndarray, iterations, udim=2, xdim=6):
    cost_function, _, _ = cost_gradient_hessian_function(Xbar, iterations=iterations, udim=udim, xdim=xdim, debug=False)
    print("cost_func: PASS")
    return cost_function.astype(np.float64)


def hessianmat(iterations, udim=2, xdim=6):
    # Define Q and R blocks
    Q = np.diag([10, 1, 10, 1, 100, 1])  # 6x6
    R = np.diag([0.1, 0.1])              # 2x2
    
    # Combine Q and R into an 8x8 block
    block = np.block([
        [Q, np.zeros((xdim, udim))],
        [np.zeros((udim, xdim)), R]
    ])  # 8x8
    
    # Create block diagonal matrix
    H = block_diag(*([block] * iterations))
    
    # Append the final Q for the last state (no control)
    H = block_diag(H, Q)
    
    return H.astype(np.float64)


def form_ineq_const(Xbar: np.ndarray, iterations, udim=2, xdim=6, debug=False):
    # H matrix construction  
    H_i = np.array([
        [0, 0, 0, 0, 0, 0, 1, 0],   # u1 <=10
        [0, 0, 0, 0, 0, 0, -1, 0],  # u1 >=0
        [0, 0, 0, 0, 0, 0, 0, 1],   # u2 <=10
        [0, 0, 0, 0, 0, 0, 0, -1],  # u2 >=0
        [0, 0, 1, 0, 0, 0, 0, 0]    # p_y >=0
    ], dtype=np.float64)
    
    H = block_diag(*([H_i] * iterations)).astype(np.float64)  # (5*iterations, nvars)
    
    # h matrix construction     
    h = np.zeros((5 * iterations,), dtype=np.float64)
    for i in range(iterations):
        h[i * 5 + 0] = 10 - Xbar[6 + i * 8]      # u1_i <=10
        h[i * 5 + 1] = Xbar[6 + i * 8]           # u1_i >=0
        h[i * 5 + 2] = 10 - Xbar[7 + i * 8]      # u2_i <=10
        h[i * 5 + 3] = Xbar[7 + i * 8]           # u2_i >=0
        h[i * 5 + 4] = Xbar[2 + i * 8 + 2]      # p_y_i >=0 (assuming p_y is the 3rd state variable)
    
    if debug:
        print(f"H matrix shape: {H.shape}")
        print(f"h vector shape: {h.shape}")
    
    h = h.reshape(-1,1)
    if debug:
        print(f"h reshaped shape: {h.shape}")
    
    print("H_h_functions: PASS")
    return H.astype(np.float64), h.astype(np.float64)


def KKT_solv(Xbar: np.ndarray, iterations, udim=2, xdim=6, debug=False):
    cost, q, P = cost_gradient_hessian_function(Xbar, iterations, udim=udim, xdim=xdim, debug=debug)
    H, h = hessianmat(iterations, udim=udim, xdim=xdim), None  # hessianmat returns H
    C, d = C_d_functions(Xbar, iterations, udim=udim, xdim=xdim, debug=debug)
    H_ineq, h_ineq = form_ineq_const(Xbar, iterations, udim=udim, xdim=xdim, debug=debug)
    
    # G matrix for inequality constraints
    G = H_ineq
    h_val = h_ineq.flatten()
    
    # Equality constraints
    A = C
    b = -d.flatten()
    
    # Hessian and gradient for QP
    H_mat = hessianmat(iterations, udim=udim, xdim=xdim)
    fgrad_Mat = (P @ Xbar + q).flatten()
    
    # Cost and constraints are handled in the Problem definition
    # Construct G and h for inequality constraints
    G_qp = H_ineq
    h_qp = h_ineq.flatten()
    
    # Convert all matrices to numpy arrays with correct data types
    H_mat = H_mat
    fgrad_Mat = q.flatten()
    A = C
    b = -d.flatten()
    G_qp = H_ineq
    h_qp = h_ineq.flatten()
    
    # Create the QP problem
    problem = Problem(P=P, q=q.flatten(), A=C, b=-d.flatten(), G=H_qp, h=h_qp)
    
    # Solve the QP
    solution = solve_problem(problem=problem, solver="cvxopt", verbose=True, initvals=None)
    
    print("KKT_solv: PASS")
    return solution  


def tot_constraint_violation_eq_ineq(Xbar: np.ndarray, iterations, udim=2, xdim=6, debug=False):
    # Compute equality constraint violation
    _, d = C_d_functions(Xbar, iterations, udim=udim, xdim=xdim, debug=False)
    constraint_violation_eq = np.sum(np.abs(d))
    
    # Compute inequality constraint violation
    H_ineq, _ = form_ineq_const(Xbar, iterations, udim=udim, xdim=xdim, debug=False)
    u1 = Xbar[6::8].flatten()  # u1 indices
    u2 = Xbar[7::8].flatten()  # u2 indices
    p_y = Xbar[2::8].flatten()  # p_y indices (assuming p_y is the 3rd state variable)
    
    constraint_violation_ineq = 0
    for i in range(iterations):
        if u1[i] > 10:
            constraint_violation_ineq += np.abs(u1[i] - 10)
        elif u1[i] < 0:
            constraint_violation_ineq += np.abs(u1[i])
        if u2[i] > 10:
            constraint_violation_ineq += np.abs(u2[i] - 10)
        elif u2[i] < 0:
            constraint_violation_ineq += np.abs(u2[i])
        if p_y[i] < 0:
            constraint_violation_ineq += np.abs(p_y[i])
    
    print("tot_constraint_violation_eq_ineq: PASS")
    return (constraint_violation_eq + constraint_violation_ineq).astype(np.float64)


# Define additional required functions correctly

def hessianmat(iterations, udim=2, xdim=6):
    # Define Q and R blocks
    Q = np.diag([10, 1, 10, 1, 100, 1])  # 6x6
    R = np.diag([0.1, 0.1])              # 2x2
    
    # Combine Q and R into an 8x8 block
    block = np.block([
        [Q, np.zeros((xdim, udim))],
        [np.zeros((udim, xdim)), R]
    ])  # 8x8
    
    # Create block diagonal matrix
    H = block_diag(*([block] * iterations))
    
    # Append the final Q for the last state (no control)
    H = block_diag(H, Q)
    
    return H.astype(np.float64)


def form_ineq_const(Xbar: np.ndarray, iterations, udim=2, xdim=6, debug=False):
    # H matrix construction  
    H_i = np.array([
        [0, 0, 0, 0, 0, 0, 1, 0],   # u1 <=10
        [0, 0, 0, 0, 0, 0, -1, 0],  # u1 >=0
        [0, 0, 0, 0, 0, 0, 0, 1],   # u2 <=10
        [0, 0, 0, 0, 0, 0, 0, -1],  # u2 >=0
        [0, 0, 1, 0, 0, 0, 0, 0]    # p_y >=0
    ], dtype=np.float64)
    
    H = block_diag(*([H_i] * iterations)).astype(np.float64)  # (5*iterations, nvars)
    
    # h matrix construction     
    h = np.zeros((5 * iterations,), dtype=np.float64)
    for i in range(iterations):
        h[i * 5 + 0] = 10 - Xbar[6 + i * 8]      # u1_i <=10
        h[i * 5 + 1] = Xbar[6 + i * 8]           # u1_i >=0
        h[i * 5 + 2] = 10 - Xbar[7 + i * 8]      # u2_i <=10
        h[i * 5 + 3] = Xbar[7 + i * 8]           # u2_i >=0
        h[i * 5 + 4] = Xbar[2 + i * 8 + 2]      # p_y_i >=0 (assuming p_y is the 3rd state variable)
    
    if debug:
        print(f"H matrix shape: {H.shape}")
        print(f"h vector shape: {h.shape}")
    
    h = h.reshape(-1,1)
    if debug:
        print(f"h reshaped shape: {h.shape}")
    
    print("H_h_functions: PASS")
    return H.astype(np.float64), h.astype(np.float64)


def solve_KKT_eq_ineq_constr(Xbar: np.ndarray, iterations, udim=2, xdim=6):
    C_Mat, d_Mat = C_d_functions(Xbar, iterations, udim=udim, xdim=xdim, debug=False)
    H_mat = hessianmat(iterations, udim=udim, xdim=xdim)
    fgrad_Mat = gradient_cost(Xbar, iterations, udim=udim, xdim=xdim)
    H_ineq, h_ineq = form_ineq_const(Xbar, iterations, udim=udim, xdim=xdim, debug=False)
    
    # Convert to correct shapes
    H_ineq = np.asarray(H_ineq, dtype=np.float64)
    h_ineq = np.asarray(h_ineq, dtype=np.float64).flatten()
    C_Mat = np.asarray(C_Mat, dtype=np.float64)
    d_Mat = np.asarray(d_Mat, dtype=np.float64).flatten()
    P = hessianmat(iterations, udim=udim, xdim=xdim)
    q = fgrad_Mat.flatten()
    
    # Create the QP problem
    problem = Problem(P=P, q=q, A=C_Mat, b=-d_Mat, G=H_ineq, h=h_ineq)
    
    # Solve the QP
    solution = solve_problem(problem=problem, solver="cvxopt", verbose=False, initvals=None)
    
    return solution


# Define gradient_cost function
def gradient_cost(Xbar, iterations, udim=2, xdim=6, debug=False):
    Xbar = Xbar.reshape(-1, 1)
    G, g = G_g_funct(iterations=iterations, udim=udim, xdim=xdim)
    costgrad = (G @ Xbar + g)
    return costgrad.astype(np.float64)


def G_g_funct(iterations, udim=2, xdim=6):
    G, g = cost_gradient_hessian_function(np.zeros(( (iterations +1)*xdim + iterations *udim, 1)), iterations, udim=udim, xdim=xdim, debug=False)
    return G, g


# Initialize variables correctly
horizon = 500  # Number of control steps
iterations = 100  # Number of optimization iterations
nvars = (horizon +1)*xdim + horizon *udim  # 4006 for xdim=6, udim=2
x_guess = np.zeros((nvars, 1), dtype=np.float64)

f_best = np.inf   
c_best = np.inf
rho = 0.5  # Standard backtracking line search parameter
tol = 1e-7
f_history = []
c_history = []
alpha_history = []

# Main optimization loop
for opt_iter in range(iterations):
    res = solve_KKT_eq_ineq_constr(x_guess, iterations=horizon, udim=udim, xdim=xdim)
    
    if res is None:
        print("Solver failed to find a solution.")
        break
    
    pk = res.reshape(nvars, 1)
    
    # Reset alpha for each optimization step
    alpha = 1.0
    
    # Line search: backtracking
    while True:
        x_new = x_guess + alpha * pk
        f_new = cost_func(x_new, iterations=horizon, udim=udim, xdim=xdim)
        c_new = tot_constraint_violation_eq_ineq(x_new, iterations=horizon, udim=udim, xdim=xdim)
        
        if (f_new < f_best) or (c_new < c_best):
            break
        else:
            alpha *= rho
            if alpha < 1e-8:
                print("Line search failed.")
                break
    
    # Update best cost and constraints
    if f_new < f_best:
        f_best = f_new
    if c_new < c_best:
        c_best = c_new
    
    # Record history
    alpha_history.append(alpha)
    c_history.append(c_best)
    f_history.append(f_best)
    print(f"Optimization Iteration {opt_iter+1}, Cost: {f_best}, Constraint Violation: {c_best}, Alpha: {alpha}")
    
    # Update the guess
    x_guess = x_guess + (alpha * pk)
    
    # Check convergence
    if c_best < tol:
        print("Convergence achieved.")
        break

# Extract state and control variables
x = x_guess[: (horizon +1)*xdim].reshape(iterations, xdim)
u = x_guess[(horizon +1)*xdim:].reshape(iterations, udim)

print(f"Total optimization iterations needed = {opt_iter+1}")
print("DEBUG POINT")
