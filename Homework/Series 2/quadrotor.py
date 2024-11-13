import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mp

# Quadrotor system matrices
m = 0.5
I = 0.1
r = 0.15
g = 9.81
dt = 0.01
length = 0.15

A = np.eye(6)
A[0,1] = dt
A[1,4] = -g * dt
A[2,3] = dt
A[4,5] = dt

B = np.zeros((6,2))
B[3,0] = dt/m
B[3,1] = dt/m
B[5,0] = length * dt/I
B[5,1] = -length * dt/I


def animate_robot(x0, u, goal):
    """
    Animates the behavior of the quadrotor based on control inputs (u).
    This version is compatible with VS Code or a regular Python script.
    """

    assert(u.shape[0] == 2)
    assert(x0.shape[0] == 6)
    N = u.shape[1] + 1
    x = np.zeros((6, N))
    x[:, 0] = x0[:, 0]

    # Simulate the system dynamics over time
    for i in range(N - 1):
        x[:, i + 1] = A @ x[:, i] + B @ u[:, i]

    # Adjusting the time steps for the animation
    min_dt = 0.1
    if dt < min_dt:
        steps = int(min_dt / dt)
        use_dt = int(np.round(min_dt * 1000))
    else:
        steps = 1
        use_dt = int(np.round(dt * 1000))

    # Downsample the state and control inputs for plotting
    plotx = x[:, ::steps]
    plotx = plotx[:, :-1]
    plotu = u[:, ::steps]

    # Create the figure and axis for the animation
    fig, ax = plt.subplots(figsize=[6, 6])
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.grid()

    list_of_lines = []

    # Create the quadrotor body and components
    line, = ax.plot([], [], 'k', lw=6)  # Main frame
    list_of_lines.append(line)
    line, = ax.plot([], [], 'b', lw=4)  # Left propeller
    list_of_lines.append(line)
    line, = ax.plot([], [], 'b', lw=4)  # Right propeller
    list_of_lines.append(line)
    line, = ax.plot([], [], 'r', lw=1)  # Left thrust
    list_of_lines.append(line)
    line, = ax.plot([], [], 'r', lw=1)  # Right thrust
    list_of_lines.append(line)

    # Plot the goal position
    ax.plot([goal[0]], [goal[1]], 'og', lw=2)

    def _animate(i):
        # Clear all lines
        for l in list_of_lines:
            l.set_data([], [])

        # Extract position and orientation
        theta = plotx[4, i]
        x = plotx[0, i]
        y = plotx[2, i]
        trans = np.array([[x, x], [y, y]])
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # Define main frame and propellers
        main_frame = np.array([[-length, length], [0, 0]])
        main_frame = rot @ main_frame + trans

        left_propeller = np.array([[-1.3 * length, -0.7 * length], [0.1, 0.1]])
        left_propeller = rot @ left_propeller + trans

        right_propeller = np.array([[1.3 * length, 0.7 * length], [0.1, 0.1]])
        right_propeller = rot @ right_propeller + trans

        left_thrust = np.array([[length, length], [0.1, 0.1 + plotu[0, i] * 0.04]])
        left_thrust = rot @ left_thrust + trans

        right_thrust = np.array([[-length, -length], [0.1, 0.1 + plotu[0, i] * 0.04]])
        right_thrust = rot @ right_thrust + trans

        # Update the quadrotor components with the new data
        list_of_lines[0].set_data(main_frame[0, :], main_frame[1, :])
        list_of_lines[1].set_data(left_propeller[0, :], left_propeller[1, :])
        list_of_lines[2].set_data(right_propeller[0, :], right_propeller[1, :])
        list_of_lines[3].set_data(left_thrust[0, :], left_thrust[1, :])
        list_of_lines[4].set_data(right_thrust[0, :], right_thrust[1, :])

        return list_of_lines

    def _init():
        return _animate(0)

    # Create the animation
    ani = FuncAnimation(fig, _animate, np.arange(0, len(plotx[0, :])),
                        interval=use_dt, blit=True, init_func=_init)

    # Show the animation in a window (or save it if desired)
    plt.show()

    # Optionally save the animation as an MP4 or GIF file
    # ani.save('quadrotor_animation.mp4', writer='ffmpeg')  # Uncomment to save as MP4
