"""
LSH Feature-Hash Analysis

This script analyzes how individual features contribute to LSH hash values
and ultimately to classification accuracy.

Pipeline:
1. Find balanced LSH configuration (precision-recall tradeoff)
2. Initialize multiple LSH instances with same config (to average over random planes)
3. For each LSH: Feature values → Hash contributions → Hash collisions → Accuracy
4. Aggregate and visualize which features consistently drive correct clustering
"""

import numpy as np
import os
import re
import argparse
from collections import Counter
from sklearn.preprocessing import StandardScaler
from lshashpy3 import LSHash
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Import from existing modules
# Python module names can't start with numbers, so we use importlib
import importlib.util
spec = importlib.util.spec_from_file_location("svm_module", "12_03_2025new_svm.py")
svm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svm_module)
load_holistic_landmarks = svm_module.load_holistic_landmarks
compute_all_features = svm_module.compute_all_features


# =============================================================================
# FEATURE NAMES (must match compute_all_features output order)
# =============================================================================
FEATURE_NAMES = [
    # Phonological features (indices 0-16)
    'max_angle',           # 0
    'sum_angle',           # 1
    'max_rotation',        # 2
    'sum_rotation',        # 3
    'aperture_change',     # 4
    'aperture_range',      # 5
    'aperture_std',        # 6
    'best_region',         # 7
    'best_dim',            # 8
    'best_signed_value',   # 9
    'body_contact_frames', # 10
    'min_body_distance',   # 11
    'body_contact_binary', # 12
    'hand_contact_frames', # 13
    'min_hand_distance',   # 14
    'hand_contact_binary', # 15
    'straightness',        # 16
    # Motion features (indices 17-25)
    'mean_velocity',       # 17
    'velocity_std',        # 18
    'max_velocity',        # 19
    'min_velocity',        # 20
    'mean_acceleration',   # 21
    'acceleration_std',    # 22
    'max_acceleration',    # 23
    'min_acceleration',    # 24
    'total_path_length'    # 25
]


# =============================================================================
# DATA LOADING
# =============================================================================
def load_and_prepare_data(data_dir):
    """Load video data and extract normalized features"""
    print("Loading data...")
    X_landmarks, y_labels, filenames = load_holistic_landmarks(data_dir)
    X_features = np.array([compute_all_features(landmarks) for landmarks in X_landmarks])
    
    # Handle invalid features
    invalid_mask = ~np.isfinite(X_features).all(axis=1)
    if invalid_mask.any():
        print(f"Warning: {invalid_mask.sum()} samples have invalid features, removing them")
        X_features = X_features[~invalid_mask]
        y_labels = y_labels[~invalid_mask]
        filenames = [f for i, f in enumerate(filenames) if not invalid_mask[i]]
    
    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_features)
    
    print(f"Loaded {len(filenames)} videos")
    print(f"Feature dimension: {X_normalized.shape[1]}")
    print(f"Labels: telic={sum(y == 'telic' for y in y_labels)}, atelic={sum(y == 'atelic' for y in y_labels)}")
    
    return X_normalized, y_labels, filenames, scaler


# =============================================================================
# BALANCED CONFIG FINDER
# =============================================================================
def compute_balance_score(precision, recall):
    """
    F1 score: harmonic mean of collision precision and recall.
    This balances hash bucket purity (precision) with coverage (recall).
    """
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def compute_collision_stats_for_config(lsh, X_normalized, y_labels):
    """
    Compute collision precision and recall for a given LSH configuration.
    This measures hash bucket quality directly.
    """
    n_samples = len(X_normalized)
    planes = np.array(lsh.uniform_planes)
    n_tables = lsh.num_hashtables
    
    # Get all hashes
    all_hashes = []
    for i in range(n_samples):
        hashes = [lsh._hash(planes[t], X_normalized[i]) for t in range(n_tables)]
        all_hashes.append(hashes)
    
    # Count collisions
    correct_collisions = 0
    incorrect_collisions = 0
    total_same_label_pairs = 0
    
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            same_label = y_labels[i] == y_labels[j]
            if same_label:
                total_same_label_pairs += 1
            
            # Check if they share any hash bucket
            shares_bucket = any(all_hashes[i][t] == all_hashes[j][t] 
                               for t in range(n_tables))
            
            if shares_bucket:
                if same_label:
                    correct_collisions += 1
                else:
                    incorrect_collisions += 1
    
    total_collisions = correct_collisions + incorrect_collisions
    precision = correct_collisions / total_collisions if total_collisions > 0 else 0
    recall = correct_collisions / total_same_label_pairs if total_same_label_pairs > 0 else 0
    
    return {
        'collision_precision': precision,
        'collision_recall': recall,
        'correct_collisions': correct_collisions,
        'incorrect_collisions': incorrect_collisions,
        'total_same_label_pairs': total_same_label_pairs
    }


def find_all_configs(X_normalized, y_labels, filenames,
                     hash_sizes=[3, 4, 5],
                     num_hashtables_list=[5, 10, 15],
                     top_k_list=[2, 4, 6, 8]):
    """
    Test all LSH configurations and return results with balance scores.
    
    Balance score is now based on collision precision & recall (hash bucket quality),
    not classification accuracy which can be misleading due to distance ranking.
    """
    input_dim = X_normalized.shape[1]
    results = []
    
    print("\n" + "="*80)
    print("TESTING LSH CONFIGURATIONS")
    print("(Balance score = F1 of collision precision & recall)")
    print("="*80)
    
    for hash_size in hash_sizes:
        for num_hashtables in num_hashtables_list:
            print(f"\nTesting: hash_size={hash_size}, num_hashtables={num_hashtables}")
            
            # Create LSH instance
            lsh = LSHash(hash_size=hash_size, input_dim=input_dim,
                         num_hashtables=num_hashtables)
            
            # Index all videos
            train_data_map = {}
            for idx in range(len(X_normalized)):
                train_data_map[filenames[idx]] = y_labels[idx]
                lsh.index(X_normalized[idx], extra_data=filenames[idx])
            
            # Compute collision precision and recall (hash bucket quality)
            collision_stats = compute_collision_stats_for_config(lsh, X_normalized, y_labels)
            collision_precision = collision_stats['collision_precision']
            collision_recall = collision_stats['collision_recall']
            
            # Query for different k values (for reference, but not used in balance score)
            for top_k in top_k_list:
                correct_predictions = 0
                same_label_collisions = []
                
                for test_idx, (test_sample, true_label) in enumerate(zip(X_normalized, y_labels)):
                    neighbors = lsh.query(test_sample, num_results=top_k)
                    
                    neighbor_labels = []
                    for (_, filename), distance in neighbors:
                        neighbor_labels.append(train_data_map[filename])
                    
                    # Majority voting
                    label_counts = Counter(neighbor_labels)
                    predicted_label = label_counts.most_common(1)[0][0]
                    
                    # Same-label collision rate (for reference)
                    same_label_count = sum(1 for label in neighbor_labels if label == true_label)
                    same_label_collisions.append(same_label_count / len(neighbor_labels))
                    
                    if predicted_label == true_label:
                        correct_predictions += 1
                
                accuracy = correct_predictions / len(X_normalized)
                knn_collision_rate = np.mean(same_label_collisions)
                
                # Balance score now uses collision precision & recall (hash quality)
                balance_score = compute_balance_score(collision_precision, collision_recall)
                
                results.append({
                    'hash_size': hash_size,
                    'num_hashtables': num_hashtables,
                    'top_k': top_k,
                    'accuracy': accuracy,
                    'knn_collision_rate': knn_collision_rate,
                    'collision_precision': collision_precision,
                    'collision_recall': collision_recall,
                    'balance_score': balance_score
                })
                
                print(f"  top_k={top_k}: acc={accuracy:.3f}, prec={collision_precision:.3f}, rec={collision_recall:.3f}, F1={balance_score:.3f}")
    
    return results


def find_balanced_config(results):
    """
    Find configuration with best balance score.
    
    Balance score = F1 of collision precision & recall.
    This measures hash bucket quality directly, unlike accuracy which
    can be inflated by distance-based ranking after bucket retrieval.
    """
    best_result = max(results, key=lambda x: x['balance_score'])
    
    print("\n" + "-"*80)
    print("BALANCED CONFIGURATION (Best Collision F1)")
    print("-"*80)
    print(f"Hash Size: {best_result['hash_size']}")
    print(f"Num Hash Tables: {best_result['num_hashtables']}")
    print(f"Top K: {best_result['top_k']}")
    print(f"Classification Accuracy: {best_result['accuracy']:.3f}")
    print(f"Collision Precision: {best_result['collision_precision']:.3f}")
    print(f"Collision Recall: {best_result['collision_recall']:.3f}")
    print(f"Balance Score (F1): {best_result['balance_score']:.3f}")
    
    # Explain what these mean
    print("\nInterpretation:")
    print(f"  - {best_result['collision_precision']*100:.1f}% of hash bucket collisions are same-class pairs")
    print(f"  - {best_result['collision_recall']*100:.1f}% of same-class pairs hash to same bucket")
    
    return best_result


# =============================================================================
# HASH CONTRIBUTION ANALYSIS
# =============================================================================
def analyze_feature_hash_contributions(lsh, X_normalized, y_labels, feature_names):
    """
    Analyze which features drive hash similarity within LSH.
    
    Returns dict with:
    - avg_hash_contribution: Average |plane * feature| across all samples/bits
    - discriminative_power: class_separation / within_class_variance
    - class_separation: |mean(telic) - mean(atelic)| per feature
    """
    n_samples = len(X_normalized)
    n_features = X_normalized.shape[1]
    n_tables = lsh.num_hashtables
    hash_size = lsh.hash_size
    
    # Extract all projection planes
    planes = np.array(lsh.uniform_planes)  # shape: (n_tables, hash_size, n_features)
    
    # Compute hash bit contributions for each sample
    # contribution magnitude = |plane[t,b,i] * feature[s,i]|
    contributions = np.zeros((n_samples, n_tables, hash_size, n_features))
    for s in range(n_samples):
        for t in range(n_tables):
            contributions[s, t] = planes[t] * X_normalized[s]  # broadcasting over hash_size
    
    # Average absolute contribution across all samples and hash bits
    avg_contribution = np.mean(np.abs(contributions), axis=(0, 1, 2))
    
    # Class-conditional analysis
    telic_mask = np.array([y == 'telic' for y in y_labels])
    atelic_mask = ~telic_mask
    
    # Within-class variance (low = consistent = good for hashing)
    telic_feature_var = np.var(X_normalized[telic_mask], axis=0)
    atelic_feature_var = np.var(X_normalized[atelic_mask], axis=0)
    within_class_var = (telic_feature_var + atelic_feature_var) / 2
    
    # Between-class separation
    telic_mean = np.mean(X_normalized[telic_mask], axis=0)
    atelic_mean = np.mean(X_normalized[atelic_mask], axis=0)
    class_separation = np.abs(telic_mean - atelic_mean)
    
    # Discriminative power: separation / within-variance
    discriminative_power = class_separation / (within_class_var + 1e-8)
    
    return {
        'avg_hash_contribution': avg_contribution,
        'telic_variance': telic_feature_var,
        'atelic_variance': atelic_feature_var,
        'within_class_var': within_class_var,
        'class_separation': class_separation,
        'discriminative_power': discriminative_power,
        'telic_mean': telic_mean,
        'atelic_mean': atelic_mean,
        'planes': planes
    }


def analyze_hash_collisions(lsh, X_normalized, y_labels, feature_names):
    """
    For pairs that hash to same bucket, analyze which features drove the collision.
    
    Key insight: Features with small differences between samples will contribute
    similarly to the hash, regardless of random plane values.
    """
    n_samples = len(X_normalized)
    n_features = X_normalized.shape[1]
    planes = np.array(lsh.uniform_planes)
    n_tables = lsh.num_hashtables
    
    # Get all hashes
    all_hashes = []
    for i in range(n_samples):
        hashes = [lsh._hash(planes[t], X_normalized[i]) for t in range(n_tables)]
        all_hashes.append(hashes)
    
    # Analyze hash collisions
    correct_collision_diffs = []  # same label pairs that share a bucket
    incorrect_collision_diffs = []  # different label pairs that share a bucket
    no_collision_same_label = []  # same label but no shared bucket
    no_collision_diff_label = []  # different label and no shared bucket
    
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            # Check if they share any hash bucket across any table
            shares_bucket = any(all_hashes[i][t] == all_hashes[j][t] 
                               for t in range(n_tables))
            
            feature_diff = np.abs(X_normalized[i] - X_normalized[j])
            same_label = y_labels[i] == y_labels[j]
            
            if shares_bucket:
                if same_label:
                    correct_collision_diffs.append(feature_diff)
                else:
                    incorrect_collision_diffs.append(feature_diff)
            else:
                if same_label:
                    no_collision_same_label.append(feature_diff)
                else:
                    no_collision_diff_label.append(feature_diff)
    
    # Convert to arrays and compute statistics
    results = {
        'n_correct_collisions': len(correct_collision_diffs),
        'n_incorrect_collisions': len(incorrect_collision_diffs),
        'n_missed_same_label': len(no_collision_same_label),
        'n_correct_separations': len(no_collision_diff_label),
    }
    
    if correct_collision_diffs:
        results['correct_collision_mean_diff'] = np.mean(correct_collision_diffs, axis=0)
        results['correct_collision_std_diff'] = np.std(correct_collision_diffs, axis=0)
    else:
        results['correct_collision_mean_diff'] = np.zeros(n_features)
        results['correct_collision_std_diff'] = np.zeros(n_features)
    
    if incorrect_collision_diffs:
        results['incorrect_collision_mean_diff'] = np.mean(incorrect_collision_diffs, axis=0)
        results['incorrect_collision_std_diff'] = np.std(incorrect_collision_diffs, axis=0)
    else:
        results['incorrect_collision_mean_diff'] = np.zeros(n_features)
        results['incorrect_collision_std_diff'] = np.zeros(n_features)
    
    # Collision precision and recall
    total_same_label_pairs = results['n_correct_collisions'] + results['n_missed_same_label']
    total_collisions = results['n_correct_collisions'] + results['n_incorrect_collisions']
    
    results['collision_precision'] = results['n_correct_collisions'] / total_collisions if total_collisions > 0 else 0
    results['collision_recall'] = results['n_correct_collisions'] / total_same_label_pairs if total_same_label_pairs > 0 else 0
    
    return results


def compute_classification_accuracy(lsh, X_normalized, y_labels, filenames, top_k):
    """Compute classification accuracy using majority voting among top-k neighbors"""
    train_data_map = {filenames[i]: y_labels[i] for i in range(len(filenames))}
    
    correct = 0
    predictions = []
    
    for test_sample, true_label in zip(X_normalized, y_labels):
        neighbors = lsh.query(test_sample, num_results=top_k)
        neighbor_labels = [train_data_map[filename] for (_, filename), _ in neighbors]
        predicted = Counter(neighbor_labels).most_common(1)[0][0]
        predictions.append(predicted)
        if predicted == true_label:
            correct += 1
    
    return correct / len(y_labels), predictions


# =============================================================================
# MULTI-LSH ANALYSIS
# =============================================================================
def run_multi_lsh_analysis(X_normalized, y_labels, filenames, config, n_instances=5):
    """
    Initialize multiple LSH instances with same config and analyze each.
    This averages over the randomness in projection plane initialization.
    """
    print("\n" + "="*80)
    print(f"MULTI-LSH ANALYSIS ({n_instances} instances)")
    print("="*80)
    print(f"Config: hash_size={config['hash_size']}, num_hashtables={config['num_hashtables']}, top_k={config['top_k']}")
    
    input_dim = X_normalized.shape[1]
    n_features = len(FEATURE_NAMES)
    
    all_results = []
    
    # Aggregated metrics across instances
    all_hash_contributions = []
    all_discriminative_power = []
    all_correct_collision_diffs = []
    all_incorrect_collision_diffs = []
    all_collision_precision = []
    all_collision_recall = []
    all_accuracies = []
    
    for instance_idx in range(n_instances):
        print(f"\n--- Instance {instance_idx + 1}/{n_instances} ---")
        
        # Create new LSH instance (random planes will differ)
        lsh = LSHash(
            hash_size=config['hash_size'],
            input_dim=input_dim,
            num_hashtables=config['num_hashtables']
        )
        
        # Index all videos
        for idx in range(len(X_normalized)):
            lsh.index(X_normalized[idx], extra_data=filenames[idx])
        
        # 1. Analyze feature-hash contributions
        print("  Analyzing feature-hash contributions...")
        contrib_results = analyze_feature_hash_contributions(
            lsh, X_normalized, y_labels, FEATURE_NAMES
        )
        all_hash_contributions.append(contrib_results['avg_hash_contribution'])
        all_discriminative_power.append(contrib_results['discriminative_power'])
        
        # 2. Analyze hash collisions
        print("  Analyzing hash collisions...")
        collision_results = analyze_hash_collisions(
            lsh, X_normalized, y_labels, FEATURE_NAMES
        )
        all_correct_collision_diffs.append(collision_results['correct_collision_mean_diff'])
        all_incorrect_collision_diffs.append(collision_results['incorrect_collision_mean_diff'])
        all_collision_precision.append(collision_results['collision_precision'])
        all_collision_recall.append(collision_results['collision_recall'])
        
        print(f"    Correct collisions: {collision_results['n_correct_collisions']}")
        print(f"    Incorrect collisions: {collision_results['n_incorrect_collisions']}")
        print(f"    Collision precision: {collision_results['collision_precision']:.3f}")
        print(f"    Collision recall: {collision_results['collision_recall']:.3f}")
        
        # 3. Compute classification accuracy
        print("  Computing classification accuracy...")
        accuracy, predictions = compute_classification_accuracy(
            lsh, X_normalized, y_labels, filenames, config['top_k']
        )
        all_accuracies.append(accuracy)
        print(f"    Accuracy: {accuracy:.3f}")
        
        # Store full results
        all_results.append({
            'lsh': lsh,
            'contrib_results': contrib_results,
            'collision_results': collision_results,
            'accuracy': accuracy,
            'predictions': predictions
        })
    
    # Aggregate results across instances
    aggregated = {
        'mean_hash_contribution': np.mean(all_hash_contributions, axis=0),
        'std_hash_contribution': np.std(all_hash_contributions, axis=0),
        'mean_discriminative_power': np.mean(all_discriminative_power, axis=0),
        'std_discriminative_power': np.std(all_discriminative_power, axis=0),
        'mean_correct_collision_diff': np.mean(all_correct_collision_diffs, axis=0),
        'mean_incorrect_collision_diff': np.mean(all_incorrect_collision_diffs, axis=0),
        'mean_collision_precision': np.mean(all_collision_precision),
        'std_collision_precision': np.std(all_collision_precision),
        'mean_collision_recall': np.mean(all_collision_recall),
        'std_collision_recall': np.std(all_collision_recall),
        'mean_accuracy': np.mean(all_accuracies),
        'std_accuracy': np.std(all_accuracies),
        'all_accuracies': all_accuracies,
        'class_separation': all_results[0]['contrib_results']['class_separation'],  # Same for all
        'within_class_var': all_results[0]['contrib_results']['within_class_var'],  # Same for all
    }
    
    return all_results, aggregated


def compute_feature_importance_score(aggregated):
    """
    Compute overall feature importance score combining multiple factors:
    1. Hash contribution (how much feature affects hash values)
    2. Discriminative power (class separation / within-class variance)
    3. Collision quality (small diff in correct collisions, large in incorrect)
    """
    n_features = len(FEATURE_NAMES)
    
    # Normalize each metric to [0, 1]
    def normalize(x):
        x = np.array(x)
        if x.max() - x.min() > 1e-8:
            return (x - x.min()) / (x.max() - x.min())
        return np.zeros_like(x)
    
    hash_contrib_norm = normalize(aggregated['mean_hash_contribution'])
    disc_power_norm = normalize(aggregated['mean_discriminative_power'])
    
    # Collision quality: we want features where correct collisions have SMALL diffs
    # and incorrect collisions have LARGE diffs
    correct_diff = aggregated['mean_correct_collision_diff']
    incorrect_diff = aggregated['mean_incorrect_collision_diff']
    
    # Ratio: incorrect_diff / (correct_diff + eps) - higher is better
    collision_quality = incorrect_diff / (correct_diff + 1e-8)
    collision_quality_norm = normalize(collision_quality)
    
    # Combined score (only hash contribution + discriminative power, NO collision quality)
    importance_score = (hash_contrib_norm + disc_power_norm) / 2
    
    return {
        'importance_score': importance_score,
        'hash_contrib_norm': hash_contrib_norm,
        'disc_power_norm': disc_power_norm,
        'collision_quality_norm': collision_quality_norm,
        'collision_quality_raw': collision_quality
    }


# =============================================================================
# VISUALIZATION
# =============================================================================
def visualize_results(aggregated, importance_scores, config, output_path):
    """Create comprehensive visualization of feature-hash analysis"""
    
    n_features = len(FEATURE_NAMES)
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    fig.suptitle(
        f'LSH Feature-Hash Analysis\n'
        f'Config: hash_size={config["hash_size"]}, num_hashtables={config["num_hashtables"]}, top_k={config["top_k"]}\n'
        f'Collision F1: {config["balance_score"]:.3f} (Prec={config["collision_precision"]:.3f}, Rec={config["collision_recall"]:.3f}) | Accuracy: {aggregated["mean_accuracy"]:.3f}',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # Sort features by importance score
    sorted_indices = np.argsort(importance_scores['importance_score'])[::-1]
    sorted_names = [FEATURE_NAMES[i] for i in sorted_indices]
    
    # ==========================================================================
    # Panel 1: Overall Feature Importance Ranking
    # ==========================================================================
    ax1 = fig.add_subplot(gs[0, :2])
    
    scores = importance_scores['importance_score'][sorted_indices]
    colors = plt.cm.viridis(np.linspace(0.8, 0.2, n_features))
    
    bars = ax1.barh(range(n_features), scores, color=colors)
    ax1.set_yticks(range(n_features))
    ax1.set_yticklabels(sorted_names, fontsize=9)
    ax1.set_xlabel('Importance Score (normalized)', fontsize=11)
    ax1.set_title('Overall Feature Importance for LSH Clustering', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=8)
    
    # ==========================================================================
    # Panel 2: Component Breakdown (stacked bar)
    # ==========================================================================
    ax2 = fig.add_subplot(gs[0, 2])
    
    x = np.arange(min(10, n_features))  # Top 10 features
    width = 0.25
    top_indices = sorted_indices[:10]
    
    hash_contrib = importance_scores['hash_contrib_norm'][top_indices]
    disc_power = importance_scores['disc_power_norm'][top_indices]
    collision_qual = importance_scores['collision_quality_norm'][top_indices]
    
    ax2.bar(x - width, hash_contrib, width, label='Hash Contribution', color='steelblue', alpha=0.8)
    ax2.bar(x, disc_power, width, label='Discriminative Power', color='coral', alpha=0.8)
    ax2.bar(x + width, collision_qual, width, label='Collision Quality', color='seagreen', alpha=0.8)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([FEATURE_NAMES[i] for i in top_indices], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Normalized Score', fontsize=10)
    ax2.set_title('Top 10 Features: Component Breakdown', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # ==========================================================================
    # Panel 3: Class Separation vs Within-Class Variance
    # ==========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    class_sep = aggregated['class_separation']
    within_var = aggregated['within_class_var']
    
    scatter = ax3.scatter(within_var, class_sep, 
                          c=importance_scores['importance_score'],
                          cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    
    # Annotate points
    for i, name in enumerate(FEATURE_NAMES):
        ax3.annotate(name, (within_var[i], class_sep[i]), 
                    fontsize=7, alpha=0.8,
                    xytext=(3, 3), textcoords='offset points')
    
    ax3.set_xlabel('Within-Class Variance', fontsize=10)
    ax3.set_ylabel('Between-Class Separation', fontsize=10)
    ax3.set_title('Feature Discriminability\n(Top-right = Best)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax3, label='Importance Score')
    ax3.grid(alpha=0.3)
    
    # ==========================================================================
    # Panel 4: Collision Analysis - Feature Differences
    # ==========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    correct_diff = aggregated['mean_correct_collision_diff']
    incorrect_diff = aggregated['mean_incorrect_collision_diff']
    
    x = np.arange(n_features)
    width = 0.35
    
    ax4.bar(x - width/2, correct_diff, width, label='Correct Collisions', color='green', alpha=0.7)
    ax4.bar(x + width/2, incorrect_diff, width, label='Incorrect Collisions', color='red', alpha=0.7)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=7)
    ax4.set_ylabel('Mean Feature Difference', fontsize=10)
    ax4.set_title('Feature Differences in Hash Collisions\n(Want: small green, large red)', 
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    
    # ==========================================================================
    # Panel 5: Collision Quality Ratio
    # ==========================================================================
    ax5 = fig.add_subplot(gs[1, 2])
    
    collision_quality = importance_scores['collision_quality_raw']
    sorted_cq_indices = np.argsort(collision_quality)[::-1]
    
    colors_cq = ['green' if cq > 1 else 'red' for cq in collision_quality[sorted_cq_indices]]
    ax5.barh(range(n_features), collision_quality[sorted_cq_indices], color=colors_cq, alpha=0.7)
    ax5.axvline(x=1, color='black', linestyle='--', linewidth=1.5, label='Threshold (1.0)')
    ax5.set_yticks(range(n_features))
    ax5.set_yticklabels([FEATURE_NAMES[i] for i in sorted_cq_indices], fontsize=8)
    ax5.set_xlabel('Collision Quality Ratio\n(incorrect_diff / correct_diff)', fontsize=10)
    ax5.set_title('Collision Quality by Feature\n(Higher = Better separation)', fontsize=12, fontweight='bold')
    ax5.invert_yaxis()
    ax5.legend(fontsize=9)
    ax5.grid(axis='x', alpha=0.3)
    
    # ==========================================================================
    # Panel 6: Hash Contribution qmap (TOP 5 features only)
    # ==========================================================================
    ax6 = fig.add_subplot(gs[2, 0])
    
    hash_contrib = aggregated['mean_hash_contribution']
    disc_power = aggregated['mean_discriminative_power']
    
    # Get top 5 feature indices by importance score
    top5_indices = np.argsort(importance_scores['importance_score'])[::-1][:5]
    
    # Custom label mapping for heatmap
    label_mapping = {
        'sum_angle': 'A2',
        'aperture_std': 'B3',
        'straightness': 'C4',
        'min_body_distance': 'D2',
        'total_path_length': 'M9',
        'velocity_std': 'M2',
        'hand_contact_binary': 'D6'
    }
    top5_names = [label_mapping.get(FEATURE_NAMES[i], FEATURE_NAMES[i]) for i in top5_indices]
    
    data_matrix = np.vstack([hash_contrib, disc_power]).T
    
    # Filter to top 5 features only
    data_matrix_top5 = data_matrix[top5_indices, :]
    
    # Normalize columns for visualization
    data_matrix_norm = (data_matrix_top5 - data_matrix_top5.min(axis=0)) / (data_matrix_top5.max(axis=0) - data_matrix_top5.min(axis=0) + 1e-8)
    
    im = ax6.imshow(data_matrix_norm, aspect=0.3, cmap='YlOrRd')
    ax6.set_yticks(range(5))
    ax6.set_yticklabels(top5_names, fontsize=14, fontweight='bold')
    ax6.set_xticks(range(2))
    ax6.set_xticklabels(['Hash\nContrib', 'Discrim.\nPower'], fontsize=11)
    ax6.set_title('Top 5 Features Metrics Heatmap (normalized)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax6, shrink=0.8)
    
    # ==========================================================================
    # Panel 7: Accuracy Distribution Across LSH Instances
    # ==========================================================================
    ax7 = fig.add_subplot(gs[2, 1])
    
    accuracies = aggregated['all_accuracies']
    ax7.bar(range(len(accuracies)), accuracies, color='steelblue', alpha=0.7)
    ax7.axhline(y=aggregated['mean_accuracy'], color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {aggregated["mean_accuracy"]:.3f}')
    ax7.fill_between([-0.5, len(accuracies)-0.5], 
                     aggregated['mean_accuracy'] - aggregated['std_accuracy'],
                     aggregated['mean_accuracy'] + aggregated['std_accuracy'],
                     color='red', alpha=0.2, label=f'±1 std: {aggregated["std_accuracy"]:.3f}')
    ax7.set_xlabel('LSH Instance', fontsize=10)
    ax7.set_ylabel('Classification Accuracy', fontsize=10)
    ax7.set_title('Accuracy Across Random LSH Initializations', fontsize=12, fontweight='bold')
    ax7.set_xticks(range(len(accuracies)))
    ax7.set_xticklabels([f'LSH {i+1}' for i in range(len(accuracies))])
    ax7.legend(fontsize=9)
    ax7.set_ylim([min(accuracies) - 0.1, 1.0])
    ax7.grid(axis='y', alpha=0.3)
    
    # ==========================================================================
    # Panel 8: Summary Statistics
    # ==========================================================================
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Top 5 features summary
    top5_indices = sorted_indices[:5]
    summary_text = "TOP 5 FEATURES FOR LSH CLUSTERING\n" + "="*40 + "\n\n"
    
    for rank, idx in enumerate(top5_indices, 1):
        name = FEATURE_NAMES[idx]
        score = importance_scores['importance_score'][idx]
        hash_c = importance_scores['hash_contrib_norm'][idx]
        disc_p = importance_scores['disc_power_norm'][idx]
        coll_q = importance_scores['collision_quality_norm'][idx]
        
        summary_text += f"{rank}. {name}\n"
        summary_text += f"   Overall Score: {score:.3f}\n"
        summary_text += f"   Hash Contribution: {hash_c:.3f}\n"
        summary_text += f"   Discriminative Power: {disc_p:.3f}\n"
        summary_text += f"   Collision Quality: {coll_q:.3f}\n\n"
    
    summary_text += "="*40 + "\n"
    summary_text += f"Collision Precision: {aggregated['mean_collision_precision']:.3f} ± {aggregated['std_collision_precision']:.3f}\n"
    summary_text += f"Collision Recall: {aggregated['mean_collision_recall']:.3f} ± {aggregated['std_collision_recall']:.3f}\n"
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def save_results_report(aggregated, importance_scores, config, output_path):
    """Save detailed text report of results"""
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LSH FEATURE-HASH ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-"*40 + "\n")
        f.write(f"Hash Size: {config['hash_size']}\n")
        f.write(f"Num Hash Tables: {config['num_hashtables']}\n")
        f.write(f"Top K: {config['top_k']}\n")
        f.write(f"Collision Precision: {config['collision_precision']:.3f}\n")
        f.write(f"Collision Recall: {config['collision_recall']:.3f}\n")
        f.write(f"Balance Score (F1): {config['balance_score']:.3f}\n\n")
        
        f.write("Config Selection Note:\n")
        f.write("  Balance score = F1(collision_precision, collision_recall)\n")
        f.write("  This measures hash bucket quality directly, not classification accuracy.\n\n")
        
        f.write("OVERALL PERFORMANCE\n")
        f.write("-"*40 + "\n")
        f.write(f"Mean Accuracy: {aggregated['mean_accuracy']:.3f} ± {aggregated['std_accuracy']:.3f}\n")
        f.write(f"Collision Precision: {aggregated['mean_collision_precision']:.3f} ± {aggregated['std_collision_precision']:.3f}\n")
        f.write(f"Collision Recall: {aggregated['mean_collision_recall']:.3f} ± {aggregated['std_collision_recall']:.3f}\n\n")
        
        f.write("FEATURE IMPORTANCE RANKING\n")
        f.write("-"*40 + "\n")
        
        sorted_indices = np.argsort(importance_scores['importance_score'])[::-1]
        
        f.write(f"{'Rank':<5} {'Feature':<25} {'Score':<8} {'Hash':<8} {'Discrim':<8} {'CollQ':<8}\n")
        f.write("-"*70 + "\n")
        
        for rank, idx in enumerate(sorted_indices, 1):
            name = FEATURE_NAMES[idx]
            score = importance_scores['importance_score'][idx]
            hash_c = importance_scores['hash_contrib_norm'][idx]
            disc_p = importance_scores['disc_power_norm'][idx]
            coll_q = importance_scores['collision_quality_norm'][idx]
            
            f.write(f"{rank:<5} {name:<25} {score:<8.3f} {hash_c:<8.3f} {disc_p:<8.3f} {coll_q:<8.3f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*80 + "\n\n")
        
        top3 = [FEATURE_NAMES[i] for i in sorted_indices[:3]]
        f.write(f"Top 3 features driving LSH clustering: {', '.join(top3)}\n\n")
        
        f.write("These features have:\n")
        f.write("  1. High contribution to hash bit computations\n")
        f.write("  2. Good class separation (telic vs atelic differ)\n")
        f.write("  3. Quality collisions (same-class pairs are similar, diff-class pairs differ)\n")
    
    print(f"Report saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='LSH Feature-Hash Analysis')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing .pkl files')
    parser.add_argument('--n_instances', type=int, default=5,
                        help='Number of LSH instances to average over (default: 5)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as data_dir parent)')
    parser.add_argument('--hash_sizes', type=int, nargs='+', default=[3, 4, 5],
                        help='Hash sizes to test (default: 3 4 5)')
    parser.add_argument('--num_hashtables', type=int, nargs='+', default=[5, 10, 15],
                        help='Numbers of hash tables to test (default: 5 10 15)')
    parser.add_argument('--top_k', type=int, nargs='+', default=[2, 4, 6, 8],
                        help='Top K values to test (default: 2 4 6 8)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.data_dir) or '.'
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("LSH FEATURE-HASH ANALYSIS")
    print("="*80)
    
    # 1. Load and prepare data
    X_normalized, y_labels, filenames, scaler = load_and_prepare_data(args.data_dir)
    
    # 2. Find balanced configuration
    all_configs = find_all_configs(
        X_normalized, y_labels, filenames,
        hash_sizes=args.hash_sizes,
        num_hashtables_list=args.num_hashtables,
        top_k_list=args.top_k
    )
    
    balanced_config = find_balanced_config(all_configs)
    
    # 3. Run multi-LSH analysis with balanced config
    all_results, aggregated = run_multi_lsh_analysis(
        X_normalized, y_labels, filenames,
        config=balanced_config,
        n_instances=args.n_instances
    )
    
    # 4. Compute feature importance scores
    print("\n" + "="*80)
    print("COMPUTING FEATURE IMPORTANCE")
    print("="*80)
    
    importance_scores = compute_feature_importance_score(aggregated)
    
    # Print top features
    sorted_indices = np.argsort(importance_scores['importance_score'])[::-1]
    print("\nTop 10 Features by Importance:")
    print("-"*50)
    for rank, idx in enumerate(sorted_indices[:10], 1):
        print(f"  {rank}. {FEATURE_NAMES[idx]}: {importance_scores['importance_score'][idx]:.3f}")
    
    # 5. Visualize results
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    viz_path = os.path.join(args.output_dir, 'lsh_feature_hash_analysis.png')
    visualize_results(aggregated, importance_scores, balanced_config, viz_path)
    
    report_path = os.path.join(args.output_dir, 'lsh_feature_hash_analysis_report.txt')
    save_results_report(aggregated, importance_scores, balanced_config, report_path)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

