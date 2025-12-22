"""
SVM vs LSH Conceptual Demonstration

Uses synthetic 2D data to clearly visualize:
- SVM: Learns ONE optimal hyperplane to separate classes
- LSH: Randomly generates multiple planes, selects best configuration
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.svm import SVC

np.random.seed(42)


def generate_synthetic_data(n_per_class=50):
    """Generate simple 2D data with two separable classes."""
    # Class A (blue) - centered at (-2, -1)
    class_a = np.random.randn(n_per_class, 2) * 0.8 + np.array([-2, -1])
    
    # Class B (red) - centered at (2, 1.5)
    class_b = np.random.randn(n_per_class, 2) * 0.8 + np.array([2, 1.5])
    
    X = np.vstack([class_a, class_b])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    
    return X, y


def draw_hyperplane(ax, w, b, xlim, color='black', linestyle='-', linewidth=2, alpha=1.0, label=None):
    """Draw a hyperplane defined by w·x + b = 0."""
    x_vals = np.linspace(xlim[0], xlim[1], 100)
    
    if abs(w[1]) > 1e-6:
        # y = -(w[0]*x + b) / w[1]
        y_vals = -(w[0] * x_vals + b) / w[1]
        ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, 
                linewidth=linewidth, alpha=alpha, label=label)
    else:
        # Vertical line
        x_val = -b / w[0]
        ax.axvline(x=x_val, color=color, linestyle=linestyle, 
                   linewidth=linewidth, alpha=alpha, label=label)


def create_visualization():
    """Create side-by-side SVM vs LSH demonstration."""
    
    # Generate synthetic data
    X, y = generate_synthetic_data(n_per_class=40)
    
    # Set up figure
    fig, axes = plt.subplots(1, 2, figsize=(24, 7))
    
    xlim = (-5, 5)
    ylim = (-4, 5)
    
    # Colors
    colors = ['#E74C3C', '#3498DB']  # Red for class 0, Blue for class 1
    
    # ==========================================================================
    # Panel 1: SVM - Learns Single Optimal Hyperplane
    # ==========================================================================
    ax1 = axes[0]
    
    # Plot data points
    for c in [0, 1]:
        mask = y == c
        ax1.scatter(X[mask, 0], X[mask, 1], c=colors[c], s=100, alpha=0.7,
                   edgecolors='black', linewidth=1, 
                   label=f'Class {"A" if c == 0 else "B"}',
                   marker='o' if c == 0 else 's')
    
    # Train SVM
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X, y)
    
    # Get hyperplane parameters
    w = svm.coef_[0]
    b = svm.intercept_[0]
    
    # Draw the learned hyperplane (decision boundary)
    draw_hyperplane(ax1, w, b, xlim, color='green', linewidth=4, 
                   label='Learned Decision Boundary')
    
    # Draw margin boundaries
    draw_hyperplane(ax1, w, b - 1, xlim, color='green', linestyle='--', 
                   linewidth=2, alpha=0.5)
    draw_hyperplane(ax1, w, b + 1, xlim, color='green', linestyle='--', 
                   linewidth=2, alpha=0.5, label='Margin')
    
    # Highlight support vectors
    sv = svm.support_vectors_
    ax1.scatter(sv[:, 0], sv[:, 1], s=250, facecolors='none', 
               edgecolors='green', linewidth=3, label='Support Vectors')
    
    
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xlabel('Feature 1', fontsize=14)
    ax1.set_ylabel('Feature 2', fontsize=14)
    ax1.set_title('SVM: Learns ONE Optimal Hyperplane', fontsize=20, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # ==========================================================================
    # Panel 2: LSH - Multiple Random Projections
    # ==========================================================================
    ax2 = axes[1]
    
    # Plot data points
    for c in [0, 1]:
        mask = y == c
        ax2.scatter(X[mask, 0], X[mask, 1], c=colors[c], s=100, alpha=0.7,
                   edgecolors='black', linewidth=1, 
                   label=f'Class {"A" if c == 0 else "B"}',
                   marker='o' if c == 0 else 's')
    
    # Generate multiple random hyperplanes
    n_random_planes = 8
    random_planes = []
    plane_colors = plt.cm.Greys(np.linspace(0.3, 0.7, n_random_planes))
    
    # Calculate separation quality for each random plane
    plane_qualities = []
    
    for i in range(n_random_planes):
        # Random direction
        angle = np.random.uniform(0, np.pi)
        w_rand = np.array([np.cos(angle), np.sin(angle)])
        b_rand = np.random.uniform(-1, 1)
        
        random_planes.append((w_rand, b_rand))
        
        # Calculate how well this plane separates the classes
        # Project points onto normal direction
        projections = X @ w_rand
        proj_class0 = projections[y == 0]
        proj_class1 = projections[y == 1]
        
        # Quality = separation between class means / pooled std
        mean_diff = abs(proj_class0.mean() - proj_class1.mean())
        pooled_std = np.sqrt((proj_class0.std()**2 + proj_class1.std()**2) / 2)
        quality = mean_diff / (pooled_std + 0.1)
        plane_qualities.append(quality)
        
        # Draw random plane (gray, dashed)
        draw_hyperplane(ax2, w_rand, b_rand, xlim, color=plane_colors[i], 
                       linestyle='--', linewidth=1.5, alpha=0.6)
    
    # Find and highlight the best random plane
    best_idx = np.argmax(plane_qualities)
    w_best, b_best = random_planes[best_idx]
    
    # Draw best configuration prominently
    draw_hyperplane(ax2, w_best, b_best, xlim, color='orange', linewidth=4,
                   label='Best Random Plane')
    
    
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_xlabel('Feature 1', fontsize=14)
    ax2.set_ylabel('Feature 2', fontsize=14)
    ax2.set_title('LSH: Generates MANY Random Hyperplanes', fontsize=20, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # ==========================================================================
    # Overall title and merged legend
    # ==========================================================================
    fig.suptitle('SVM vs LSH: Learned vs Random Hyperplane Approaches', 
                fontsize=22, fontweight='bold', y=1.02)
    
    # Get handles and labels from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    # Merge handles and labels, removing duplicates based on label
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    
    # Remove duplicates while preserving order
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(all_handles, all_labels):
        if label not in seen:
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
    
    # Adjust layout - use wspace to control spacing between subplots, and match left margin
    plt.subplots_adjust(left=0.20, right=0.95, wspace=0.1)
    
    # Create merged legend on the left side outside both graphs
    # Position it so the spacing between legend and SVM matches spacing between SVM and LSH
    leg = fig.legend(unique_handles, unique_labels, loc='center left', 
                    bbox_to_anchor=(0.02, 0.5), fontsize=20, frameon=True)
    
    plt.tight_layout(rect=[0.20, 0, 1, 0.98])
    
    output_path = 'svm_vs_lsh_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to: {output_path}")
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("KEY DIFFERENCES")
    print("="*60)
    print("\nSVM (Support Vector Machine):")
    print("  • Learns ONE optimal hyperplane")
    print("  • Uses optimization (gradient descent)")
    print("  • Maximizes margin between classes")
    print("  • Computationally expensive for large datasets")
    print("\nLSH (Locality Sensitive Hashing):")
    print("  • Generates MANY random hyperplanes")
    print("  • No learning/optimization")
    print("  • Selects best configuration from random samples")
    print("  • Fast and scalable for large datasets")
    print("="*60)


if __name__ == "__main__":
    create_visualization()

