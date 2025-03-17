import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

class ScratchDecisionTreeClassifier():
    """
    [Problem 7] Creating a depth 2 decision tree classifier class
    [Problem 8] Creating a decision tree classifier class with unlimited depth
    """
    def __init__(self, max_depth=None, verbose=False):
        self.verbose = verbose
        self.max_depth = max_depth
        self.tree = None

    def gini_impurity(self, y):
        """
        [Problem 1] Compute Gini impurity
        """
        unique_classes, counts = np.unique(y, return_counts=True)
        N = len(y)
        return 1 - sum((count / N) ** 2 for count in counts)

    def information_gain(self, y_parent, y_left, y_right):
        """
        [Problem 2] Compute Information Gain
        """
        N_parent = len(y_parent)
        IG = self.gini_impurity(y_parent) - (len(y_left) / N_parent) * self.gini_impurity(y_left) - (len(y_right) / N_parent) * self.gini_impurity(y_right)
        return IG

    def fit(self, X, y, depth=0):
        """
        [Problem 3] Training a Decision Tree Classifier
        """
        if self.max_depth is not None and depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.argmax(np.bincount(y))
        
        best_gain = 0
        best_threshold = None
        best_feature = None
        best_splits = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                IG = self.information_gain(y, y[left_mask], y[right_mask])
                if IG > best_gain:
                    best_gain = IG
                    best_threshold = threshold
                    best_feature = feature
                    best_splits = (X[left_mask], y[left_mask], X[right_mask], y[right_mask])
        
        if best_gain == 0:
            return np.argmax(np.bincount(y))
        
        left_tree = self.fit(best_splits[0], best_splits[1], depth + 1)
        right_tree = self.fit(best_splits[2], best_splits[3], depth + 1)
        
        return (best_feature, best_threshold, left_tree, right_tree)

    def predict_sample(self, sample, node):
        """
        [Problem 4] Implementing the Estimation Mechanism
        """
        if not isinstance(node, tuple):
            return node
        feature, threshold, left, right = node
        return self.predict_sample(sample, left if sample[feature] < threshold else right)

    def predict(self, X):
        return np.array([self.predict_sample(sample, self.tree) for sample in X])

    def plot_decision_boundary(self, X, y):
        """
        [Problem 6] Visualization of Decision Regions
        """
        if X.shape[1] != 2:
            raise ValueError("Plotting is only supported for 2D data")
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        predictions = self.predict(grid_points).reshape(xx.shape)
        
        plt.contourf(xx, yy, predictions, alpha=0.5)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
        plt.title(f"Decision Tree Decision Boundary (Depth {self.max_depth})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

# [Problem 5] Learning and Estimation
np.random.seed(42)
X_test = np.random.rand(50, 2) * 10
y_test = (X_test[:, 0] + X_test[:, 1] > 10).astype(int)

dt = ScratchDecisionTreeClassifier(max_depth=2)
dt.tree = dt.fit(X_test, y_test)
dt.plot_decision_boundary(X_test, y_test)

# Evaluation with Scikit-Learn
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X_test, y_test)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
