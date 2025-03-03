import numpy as np
import matplotlib.pyplot as plt

maths = np.array([92, 56, 88, 70, 80, 49, 65, 35, 66, 67])  # Maths scores
cse = np.array([98, 68, 81, 80, 83, 52, 66, 30, 68, 73])  # CSE scores

#Mean squared error function
def msqe(y, predictedy):
    return np.mean((y - predictedy) ** 2)

# Linear Search to find best m
searchspace = np.linspace(-2, 3, 50)
errlist = [msqe(cse, m * maths) for m in searchspace]
lowestm = searchspace[np.argmin(errlist)]
print(f"Best m found via Linear Search: {lowestm:.4f}")

#Gradient Descent
def gradient_descent(x, y, lr=0.00001, iterations=100):
    m = np.random.randn() 
    n = len(x)
    mselist = []
    mval = []
    converged_pt = iterations - 1  # Default to last iteration

    for i in range(iterations):
        predictedy = m * x
        mse = msqe(y, predictedy)
        mselist.append(mse)
        grad = (-2 / n) * np.sum(x * (y - predictedy))
        newm = m - lr * grad
        mval.append(newm)

        #Convergence check (if change in m is very small)
        if abs(newm - m) < 1e-6:
            print(f"Converged at iteration {i+1}")
            converged_pt = i
            break
        m = newm 

    return m, mselist, mval, converged_pt

bestm, mselist, mval, convergence_pt = gradient_descent(maths, cse)
print(f"Best m found via Gradient Descent: {bestm:.4f}")

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: MSE vs m (Linear Search)
axs[0].plot(searchspace, errlist, label="MSE vs m", color='blue')
axs[0].axvline(lowestm, color='r', linestyle="--", label=f"Best m (Linear Search) = {lowestm:.4f}")
axs[0].set_xlabel("m values")
axs[0].set_ylabel("MSE")
axs[0].set_title("Finding Best m using Linear Search")
axs[0].legend()
axs[0].grid()

# Plot 2: MSE vs Iterations (Gradient Descent)
axs[1].plot(range(len(mselist)), mselist, color='green', label="Gradient Descent MSE")
axs[1].scatter(convergence_pt, mselist[convergence_pt], color='red', zorder=3, label="Convergence Point")
axs[1].set_xlabel("iterations")
axs[1].set_ylabel("MSE")
axs[1].set_title("Gradient Descent Progression")
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()
