"""
SVM vs LSH Comparison Visualization using t-SNE

This script demonstrates the fundamental difference between:
- SVM: Learns ONE optimal hyperplane to separate classes
- LSH: Randomly generates MULTIPLE planes, finds best configuration

Visualization approach:
1. t-SNE to project high-dimensional features to 2D
2. SVM panel: Shows learned decision boundary (single plane concept)
3. LSH panel: Shows multiple random projection directions and hash buckets
"""

import numpy as np
import os
import importlib.util
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lshashpy3 import LSHash
import argparse

# Import from existing modules
spec = importlib.util.spec_from_file_location("svm_module", "12_03_2025new_svm.py")
svm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svm_module)
load_holistic_landmarks = svm_module.load_holistic_landmarks
compute_all_features = svm_module.compute_all_features

FEATURE_NAMES = [
    'max_angle', 'sum_angle', 'max_rotation', 'sum_rotation',
    'aperture_change', 'aperture_range', 'aperture_std',
    'best_region', 'best_dim', 'best_signed_value',
    'body_contact_frames', 'min_body_distance', 'body_contact_binary',
    'hand_contact_frames', 'min_hand_distance', 'hand_contact_binary',
    'straightness',
    'mean_velocity', 'velocity_std', 'max_velocity', 'min_velocity',
    'mean_acceleration', 'acceleration_std', 'max_acceleration', 
    'min_acceleration', 'total_path_length'
]


def create_svm_lsh_comparison(X_features, y_labels, output_path):
    """
    Create side-by-side comparison of SVM vs LSH approaches.
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    # Convert labels to numeric
    y_numeric = np.array([0 if label == 'atelic' else 1 for label in y_labels])
    
    # ==========================================================================
    # t-SNE projection for visualization
    # ==========================================================================
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # ==========================================================================
    # Train SVM
    # ==========================================================================
    print("Training SVM classifier...")
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X_scaled, y_numeric)
    svm_predictions = svm.predict(X_scaled)
    svm_accuracy = np.mean(svm_predictions == y_numeric)
    print(f"  SVM Accuracy: {svm_accuracy:.3f}")
    
    # Get SVM decision function values for coloring intensity
    svm_decision = svm.decision_function(X_scaled)
    
    # ==========================================================================
    # LSH with multiple random projections
    # ==========================================================================
    print("Running LSH with multiple configurations...")
    
    # Try different LSH configurations
    lsh_configs = []
    n_random_instances = 5
    
    for instance in range(n_random_instances):
        lsh = LSHash(hash_size=8, input_dim=X_scaled.shape[1], num_hashtables=4)
        
        # Index all samples
        for idx in range(len(X_scaled)):
            lsh.index(X_scaled[idx], extra_data=idx)
        
        # Get hash buckets for each sample
        hash_buckets = []
        for idx in range(len(X_scaled)):
            # Query to get bucket info
            results = lsh.query(X_scaled[idx], num_results=len(X_scaled))
            hash_buckets.append(len(results))
        
        # Compute accuracy based on nearest neighbor in same bucket
        correct = 0
        for idx in range(len(X_scaled)):
            results = lsh.query(X_scaled[idx], num_results=2)  # Get self + nearest
            if len(results) > 1:
                # Get the nearest neighbor (not self)
                neighbor_idx = results[1][0][1]
                if y_numeric[idx] == y_numeric[neighbor_idx]:
                    correct += 1
            else:
                # Only found self, random guess
                correct += 0.5
        
        accuracy = correct / len(X_scaled)
        lsh_configs.append({
            'instance': instance,
            'accuracy': accuracy,
            'lsh': lsh,
            'planes': np.array(lsh.uniform_planes)
        })
    
    # Find best LSH configuration
    best_lsh = max(lsh_configs, key=lambda x: x['accuracy'])
    print(f"  Best LSH Accuracy: {best_lsh['accuracy']:.3f}")
    
    # Get LSH bucket assignments for best config
    lsh_buckets = {}
    for idx in range(len(X_scaled)):
        for table_idx, table in enumerate(best_lsh['lsh'].hash_tables):
            for bucket_key, items in table.items():
                for item in items:
                    if item[1] == idx:
                        if idx not in lsh_buckets:
                            lsh_buckets[idx] = []
                        lsh_buckets[idx].append(bucket_key)
    
    # ==========================================================================
    # Create visualization
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Colors
    colors_true = ['#E74C3C' if y == 0 else '#3498DB' for y in y_numeric]  # Red=atelic, Blue=telic
    
    # ==========================================================================
    # Panel 1: SVM - Single Learned Hyperplane
    # ==========================================================================
    ax1 = axes[0]
    
    # Plot points colored by true class
    scatter1 = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                           c=svm_decision, cmap='RdBu', 
                           s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Mark misclassified points
    misclassified = svm_predictions != y_numeric
    ax1.scatter(X_tsne[misclassified, 0], X_tsne[misclassified, 1],
                s=150, facecolors='none', edgecolors='black', linewidth=2,
                label=f'Misclassified ({misclassified.sum()})')
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('SVM Decision Value\n(← Atelic | Telic →)', fontsize=10)
    
    # Draw conceptual decision boundary line through t-SNE space
    # Use PCA on t-SNE coords to find main axis, then draw perpendicular
    pca_tsne = PCA(n_components=2)
    pca_tsne.fit(X_tsne)
    
    # Find the direction that best separates classes in t-SNE space
    telic_center = X_tsne[y_numeric == 1].mean(axis=0)
    atelic_center = X_tsne[y_numeric == 0].mean(axis=0)
    separation_dir = telic_center - atelic_center
    separation_dir = separation_dir / np.linalg.norm(separation_dir)
    
    # Draw decision boundary perpendicular to separation direction
    midpoint = (telic_center + atelic_center) / 2
    perp_dir = np.array([-separation_dir[1], separation_dir[0]])
    
    line_length = 40
    p1 = midpoint - line_length * perp_dir
    p2 = midpoint + line_length * perp_dir
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=3, 
             label='Learned Decision Boundary')
    
    # Add arrow showing "learning" direction
    ax1.annotate('', xy=midpoint + 5*separation_dir, xytext=midpoint,
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax1.text(midpoint[0] + 6*separation_dir[0], midpoint[1] + 6*separation_dir[1], 
             'Optimized\nDirection', fontsize=9, color='green', ha='center')
    
    ax1.set_title(f'SVM: Single Learned Hyperplane\nAccuracy: {svm_accuracy:.1%}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    
    # Add text box explaining SVM
    textstr = 'SVM learns ONE optimal\nhyperplane that maximizes\nmargin between classes'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # ==========================================================================
    # Panel 2: LSH - Multiple Random Projections
    # ==========================================================================
    ax2 = axes[1]
    
    # Plot points colored by true class
    scatter2 = ax2.scatter(X_tsne[y_numeric == 0, 0], X_tsne[y_numeric == 0, 1],
                           c='#E74C3C', s=80, alpha=0.7, marker='o',
                           edgecolors='black', linewidth=0.5, label='Atelic')
    scatter3 = ax2.scatter(X_tsne[y_numeric == 1, 0], X_tsne[y_numeric == 1, 1],
                           c='#3498DB', s=80, alpha=0.7, marker='s',
                           edgecolors='black', linewidth=0.5, label='Telic')
    
    # Draw multiple random projection directions
    center = X_tsne.mean(axis=0)
    np.random.seed(42)
    
    # Generate several random directions to show LSH concept
    n_planes_to_show = 6
    colors_planes = plt.cm.Set2(np.linspace(0, 1, n_planes_to_show))
    
    for i in range(n_planes_to_show):
        # Random direction
        angle = np.random.uniform(0, 2*np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        # Draw the projection line (hyperplane is perpendicular)
        line_length = 35
        p1 = center - line_length * direction
        p2 = center + line_length * direction
        
        # Draw dashed line for random plane
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', 
                 color=colors_planes[i], linewidth=2, alpha=0.6)
        
        # Draw perpendicular (the actual hyperplane direction)
        perp = np.array([-direction[1], direction[0]])
        hp1 = center - 15 * perp
        hp2 = center + 15 * perp
        ax2.plot([hp1[0], hp2[0]], [hp1[1], hp2[1]], '-', 
                 color=colors_planes[i], linewidth=1.5, alpha=0.4)
    
    # Highlight the "selected" best configuration
    best_angle = np.arctan2(separation_dir[1], separation_dir[0]) + np.pi/2
    best_dir = np.array([np.cos(best_angle), np.sin(best_angle)])
    p1 = center - 40 * best_dir
    p2 = center + 40 * best_dir
    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=3, 
             label='Best Configuration Selected')
    
    ax2.set_title(f'LSH: Multiple Random Projections\nBest Config Accuracy: {best_lsh["accuracy"]:.1%}', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    
    # Add text box explaining LSH
    textstr = f'LSH generates {n_planes_to_show}+ RANDOM\nhyperplanes, then selects\nthe best configuration'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Add annotation showing "random" generation
    ax2.annotate('Random\nPlanes', xy=(center[0]-20, center[1]+15), fontsize=9,
                 color='gray', style='italic', ha='center')
    
    # ==========================================================================
    # Overall title and layout
    # ==========================================================================
    fig.suptitle('SVM vs LSH: Learning vs Random Projection Approaches\n' +
                 '(t-SNE visualization of ASL Telicity features)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()
    
    return svm_accuracy, best_lsh['accuracy']


def main():
    parser = argparse.ArgumentParser(description='SVM vs LSH Comparison Visualization')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing .pkl landmark files')
    parser.add_argument('--output', type=str, default='svm_vs_lsh_comparison.png',
                        help='Output path for visualization')
    args = parser.parse_args()
    
    # Load data
    print("="*60)
    print("SVM vs LSH COMPARISON")
    print("="*60)
    
    print("\nLoading data...")
    X_raw, y_labels, filenames = load_holistic_landmarks(args.data_dir)
    print(f"Loaded {len(X_raw)} samples")
    
    # Compute features
    print("\nComputing features...")
    X_features = np.array([compute_all_features(landmarks) for landmarks in X_raw])
    
    # Remove invalid samples
    valid_mask = np.isfinite(X_features).all(axis=1)
    X_features = X_features[valid_mask]
    y_labels = np.array(y_labels)[valid_mask]
    print(f"Valid samples: {len(X_features)}")
    
    # Create comparison visualization
    print("\nGenerating comparison visualization...")
    svm_acc, lsh_acc = create_svm_lsh_comparison(X_features, y_labels, args.output)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"SVM Accuracy:  {svm_acc:.1%} (learned single optimal plane)")
    print(f"LSH Accuracy:  {lsh_acc:.1%} (best of random configurations)")
    print("\nKey Difference:")
    print("  • SVM: Optimizes ONE hyperplane through gradient descent")
    print("  • LSH: Generates MANY random hyperplanes, selects best config")
    print("="*60)


if __name__ == "__main__":
    main()



