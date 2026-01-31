import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def run_knn_analysis(data_path, k_value=7, test_size=0.2, random_state=0):
    """
    Run KNN classification and visualize results for a given k value.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file
    k_value : int
        Number of neighbors for KNN (default=7)
    test_size : float
        Proportion of dataset for testing (default=0.2)
    random_state : int
        Random seed for reproducibility (default=0)
    
    Returns:
    --------
    dict : Dictionary containing model, accuracy, and predictions
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Encode Gender
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    
    # Prepare features and target
    X = df[['Gender', 'Age', 'EstimatedSalary']]
    y = df['Purchased']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=axes[0], 
                cbar_kws={'label': 'Count'})
    axes[0].set_title(f'Confusion Matrix (k={k_value})\nAccuracy: {accuracy:.2%}', 
                      fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Predicted', fontweight='bold')
    axes[0].set_ylabel('Actual', fontweight='bold')
    axes[0].set_xticklabels(['No Purchase', 'Purchase'])
    axes[0].set_yticklabels(['No Purchase', 'Purchase'])
    
    # Plot 2: Decision Boundary
    plot_decision_boundary(X_train, y_train, knn, scaler, axes[1], k_value)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"KNN Analysis Results (k={k_value})")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Confusion Matrix:")
    print(f"  True Negatives:  {cm[0,0]}")
    print(f"  False Positives: {cm[0,1]}")
    print(f"  False Negatives: {cm[1,0]}")
    print(f"  True Positives:  {cm[1,1]}")
    print(f"{'='*50}\n")
    
    return {
        'model': knn,
        'scaler': scaler,
        'accuracy': accuracy,
        'predictions': y_pred,
        'y_test': y_test,
        'confusion_matrix': cm
    }


def plot_decision_boundary(X_train, y_train, model, scaler, ax, k_value):
    """Plot decision boundary for the KNN model (Age vs Salary only)."""
    h = 1000  # step size in the mesh
    x_min, x_max = X_train['Age'].min() - 5, X_train['Age'].max() + 5
    y_min, y_max = X_train['EstimatedSalary'].min() - 5000, X_train['EstimatedSalary'].max() + 5000
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/100),
                         np.arange(y_min, y_max, (y_max-y_min)/100))
    
    # Use median gender value for visualization
    median_gender = X_train['Gender'].median()
    grid_points = np.c_[np.full(xx.ravel().shape, median_gender), xx.ravel(), yy.ravel()]
    
    Z = model.predict(scaler.transform(grid_points))
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')
    
    scatter = ax.scatter(X_train['Age'], X_train['EstimatedSalary'], 
                        c=y_train, cmap='RdYlGn', edgecolors='black', 
                        s=50, alpha=0.7)
    
    ax.set_xlabel('Age', fontweight='bold')
    ax.set_ylabel('Estimated Salary', fontweight='bold')
    ax.set_title(f'Decision Boundary (k={k_value})', fontweight='bold', fontsize=12)
    ax.legend(*scatter.legend_elements(), title='Purchased', loc='upper left')


def compare_k_values(data_path, k_values=[3, 5, 7, 9, 11, 15]):
    """
    Compare model performance across different k values.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file
    k_values : list
        List of k values to test (default=[3, 5, 7, 9, 11, 15])
    """
    accuracies = []
    
    for k in k_values:
        print(f"\nTesting k={k}...")
        result = run_knn_analysis(data_path, k_value=k)
        accuracies.append(result['accuracy'])
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('k (Number of Neighbors)', fontweight='bold', fontsize=12)
    plt.ylabel('Accuracy', fontweight='bold', fontsize=12)
    plt.title('Model Accuracy vs k Value', fontweight='bold', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    
    # Add value labels on points
    for k, acc in zip(k_values, accuracies):
        plt.annotate(f'{acc:.2%}', (k, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison summary
    print(f"\n{'='*50}")
    print("K-Value Comparison Summary")
    print(f"{'='*50}")
    for k, acc in zip(k_values, accuracies):
        print(f"k={k:2d}: {acc:.2%}")
    print(f"\nBest k: {k_values[accuracies.index(max(accuracies))]} "
          f"(Accuracy: {max(accuracies):.2%})")
    print(f"{'='*50}\n")


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Single k value analysis
    result = run_knn_analysis('Social_Network_Ads.csv', k_value=7)
    
    # Compare multiple k values
    # compare_k_values('Social_Network_Ads.csv', k_values=[3, 5, 7, 9, 11, 13, 15])
