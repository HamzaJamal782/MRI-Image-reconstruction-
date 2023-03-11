import matplotlib.pyplot as plt
import numpy as np

# Define the function
def f(x):
    return 2*np.sin(x)

# Define the x-values for the interval [0, 100]
x = np.linspace(0, 100, 1000)

# Define the x-values for the interval [0, 5]
x_interval = np.linspace(0, 5, 100)

# Define the y-values for the function on the interval [0, 5]
y_interval = f(x_interval)

# Plot the function on the interval [0, 100], but only display the portion corresponding to the interval [0, 5]
plt.plot(x, f(x), label='f(x)')
plt.plot(x_interval, y_interval, label='f(x) on [0, 5]', linewidth=3)

# Set the x-axis and y-axis labels
plt.xlabel('x')
plt.ylabel('amplitude')

# Set the title of the graph
plt.title('Graph of f(x) = 2sin(x) on [0, 100], but only displaying [0, 5]')

# Set the legend for the graph
plt.legend()

# Set the x-axis limits to [0, 5] only
plt.xlim(0, 5)

# Show the graph
plt.show()
