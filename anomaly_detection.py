import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Get X dataset
X = np.load(r'C:\0.Kangsan_Kim\Work\Engineering\Machine Learning\anomaly detection\Files\Files\data\X_part1.npy')
# Get Y dataset
Y = np.load(r'C:\0.Kangsan_Kim\Work\Engineering\Machine Learning\anomaly detection\Files\Files\data\y_val_part1.npy')

# ----------  Data exploration start  ----------
# Print the shape of datasets.
print(f'The shape of X is: {X.shape}')
print(f'The shape of Y is: {Y.shape}')

# Print the first 5 values of datasets.
print(f'The first 5 datapoints of X is:\n{X[:5, :]}')
print(f'The first 5 datapoints of Y is: \n{Y[:5,]}')

# Check if there is any anomaly in dataset Y?
if np.any(Y == 1):
    print("There is at least 1 anomaly in Y.")
else:
    print("There is no anomaly in Y.")

# Visually show X dataset
plt.scatter(X[:,0], X[:, 1], marker='x', c='b')
plt.title('Visual representation of dataset X')
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()

# Split the dataset X, Y for training and Cross validation, 8:2 ratio.
X_train, X_cv, Y_train, Y_cv = train_test_split(X, Y, test_size=0.2)
# Show the ratio of X_train and X_cv dataset size
print(f"The ratio of X_train and X_cv is: {round((X_train.shape[0] / X.shape[0]) * 100, 2)} : {round((X_cv.shape[0]/X.shape[0]) * 100, 2)}")

# ----------  Data exploration finished  ----------


# Calculate mean, variance of X(m, n). Use this along with X in PDF.
def estimate_gaussian(X):
    mu = np.mean(X, axis=0)
    var = np.mean((X - mu) ** 2, axis=0)
    return mu, var


# Calculate Probability of each datapoint using Probability Density Function(PDF).
def PDF(X, mu, var):
    probability = np.prod(1/np.sqrt(2*np.pi*var) * np.exp(-(X-mu)**2/(2*var)), axis=1)
    return probability


# Tune epsilon(threshold value) using F1 score
def select_threshold(p, y_train):
    best_epsilon = 0
    best_f1 = 0

    step_size = (np.max(p) - np.min(p)) / 1000
    for epsilon in np.arange(np.min(p), np.max(p), step_size):
        # Boolean array of predictions based on epsilon
        prediction = p < epsilon
        tp = np.sum((prediction == 1) & (y_train == 1))
        fp = np.sum((prediction == 1) & (y_train == 0))
        fn = np.sum((prediction == 0) & (y_train == 1))
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        # Update best_f1 and best_epsilon
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1


# ---------- Use the function, data to detect anomaly ----------

# Estimate Gaussian distribution parameters (mu, var) for the training data
mu, var = estimate_gaussian(X_train)
print(f"mu: {mu}\nvar: {var}")

# Calculate the probability for each point in the training set
p = PDF(X_train, mu, var)

# Select the best epsilon based on F1 score
epsilon, best_f1 = select_threshold(p, Y_train)

# Show anomalies using visualization
plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b')
plt.scatter(X_train[p < epsilon, 0], X_train[p < epsilon, 1], marker='o', facecolor='none', edgecolor='r', s=70)
plt.title("The anomalies (outliers)")
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.axis([0, 30, 0, 30])  # Adjust axis limits based on your dataset
plt.show()




