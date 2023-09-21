
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the equations of motion for the double pendulum
def double_pendulum(y, t, m1, m2, l1, l2, g):
    # Unpack the variables
    theta1, theta2, p1, p2 = y

    # Equations of motion
    c = np.cos(theta1 - theta2)
    s = np.sin(theta1 - theta2)
    theta1_dot = (l2 * p1 - l1 * p2 * c) / (l1**2 * l2 * (m1 + m2 * s**2))
    theta2_dot = ((m1 + m2) * l1 * p2 - m2 * l2 * p1 * c) / (m2 * l1 * l2**2 * (m1 + m2 * s**2))
    p1_dot = -m2 * l1 * l2 * theta2_dot * p2 * s - (m1 + m2) * g * l1 * np.sin(theta1)
    p2_dot = m2 * l1 * l2 * theta1_dot * p1 * s - m2 * g * l2 * np.sin(theta2)

    return [theta1_dot, theta2_dot, p1_dot, p2_dot]

# Set up initial conditions and parameters
m1 = 1.0  # Mass of the first pendulum bob
m2 = 1.0  # Mass of the second pendulum bob
l1 = 1.0  # Length of the first pendulum arm
l2 = 1.0  # Length of the second pendulum arm
g = 9.81  # Acceleration due to gravity

# Initial conditions [theta1, theta2, p1, p2]
initial_conditions = [np.pi/4, np.pi/4, 0, 0]

# Time grid
t = np.linspace(0, 20, 1000)

# Solve the equations of motion numerically
solution = odeint(double_pendulum, initial_conditions, t, args=(m1, m2, l1, l2, g))

# Extract the results
theta1, theta2, _, _ = solution.T

# Plot the trajectory of both pendulum bobs
plt.figure(figsize=(8, 6))
plt.plot(l1 * np.sin(theta1), -l1 * np.cos(theta1), label='Pendulum 1')
plt.plot(l1 * np.sin(theta1) + l2 * np.sin(theta2), -l1 * np.cos(theta1) - l2 * np.cos(theta2), label='Pendulum 2')
plt.title('Double Pendulum Trajectory')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid()
plt.show()
