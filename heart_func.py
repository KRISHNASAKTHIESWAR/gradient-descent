import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(15, 5)) 

# 1st heart function -- Parametric heart
def heart_function_1(t):
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    return x, y

# Generate values for 1st heart function
t = np.linspace(0, 2 * np.pi, 1000)
x, y = heart_function_1(t)

# 1st heart plot
axs[0].plot(x, y, color='red')
axs[0].set_title("Parametric Heart Curve")
axs[0].set_xlabel("X-axis")
axs[0].set_ylabel("Y-axis")
axs[0].axis("equal")
axs[0].grid(True)

#------------------------------------------------------------

# 2nd heart function -- Implicit heart
def heart_function_2(x, y):
    return (x**2 + y**2 - 1)**3 - x**2 * y**3

# Generate values for 2nd heart
x = np.linspace(-1.5, 1.5, 400)
y = np.linspace(-1.5, 1.5, 400)
X, Y = np.meshgrid(x, y)

Z = heart_function_2(X, Y)

# 2nd heart plot
axs[1].contour(X, Y, Z, levels=[0], colors='red')
axs[1].set_title("Implicit Heart Curve")
axs[1].set_xlabel("X-axis")
axs[1].set_ylabel("Y-axis")
axs[1].axis("equal")
axs[1].grid(True)


plt.tight_layout()
plt.show()
