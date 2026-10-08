import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

def plt_house_x(X, y):
    """Plot house prices against size."""
    plt.scatter(X, y, marker='x', c='red', label='Actual Data')
    plt.xlabel('Size (1000 sqft)')
    plt.ylabel('Price (1000s of dollars)')
    plt.title('Housing Prices')
    plt.legend()
    plt.show()

def plt_contour_wgrad(x, y, hist, w_range=1, b_range=100, contours=15):
    """Plot cost surface contour with gradient descent path."""
    w = np.linspace(-w_range, w_range, 20)
    b = np.linspace(-b_range, b_range, 20)
    W, B = np.meshgrid(w, b)
    
    Z = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Z[i,j] = np.mean((W[i,j]*x + B[i,j] - y)**2)/2
            
    plt.contour(W, B, Z, contours)
    plt.plot([p[0] for p in hist], [p[1] for p in hist], 'r.-')
    plt.xlabel('w')
    plt.ylabel('b')
    plt.title('Gradient Descent Path on Cost Surface')
    plt.show()

def plt_divergence(x, y, w, b):
    """Plot diverging gradient descent."""
    plt.scatter(x, y, marker='x', c='red', label='Actual Data')
    plt.plot(x, w*x + b, 'b-', label='Prediction')
    plt.xlabel('Size (1000 sqft)')
    plt.ylabel('Price (1000s of dollars)')
    plt.title('Diverging Gradient Descent')
    plt.legend()
    plt.show()

def plt_gradients(x, y, compute_cost, compute_gradient):
    """Plot gradients using compute_cost and compute_gradient functions."""
    w = 0
    b = 0
    
    # Calculate current cost and gradient
    cost = compute_cost(x, y, w, b)
    dj_dw, dj_db = compute_gradient(x, y, w, b)
    
    # Plot the data points
    plt.scatter(x, y, marker='x', c='red', label='Actual Data')
    
    # Plot the prediction line
    y_pred = w * x + b
    plt.plot(x, y_pred, 'b-', label='Current Prediction')
    
    # Add arrows to show gradients
    plt.quiver(x, y_pred, -dj_dw, -dj_db, angles='xy', scale_units='xy', scale=1)
    
    plt.xlabel('Size (1000 sqft)')
    plt.ylabel('Price (1000s of dollars)')
    plt.title('Gradient Visualization')
    plt.legend()
    plt.show()