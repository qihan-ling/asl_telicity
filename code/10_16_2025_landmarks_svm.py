import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def load_holistic_landmarks(data_dir):
    """
    Loads all .pkl files in the directory.
    Labels are inferred from filenames: telic_*.pkl → label 'telic', atelic_*.pkl → label 'atelic'.
    """
    X, y, filenames = [], [], []
    for file in os.listdir(data_dir):
        if file.endswith(".pkl"):
            file_path = os.path.join(data_dir, file)
            with open(file_path, 'rb') as f:
                landmarks_sequence = pickle.load(f)
            X.append(landmarks_sequence)
            filenames.append(file)

            if "atelic" in file.lower():
                y.append("atelic")
            elif "telic" in file.lower():
                y.append("telic")
            else:
                raise ValueError(f"File {file} does not contain telic/atelic label")

    return X, np.array(y), filenames


def normalize_landmarks(landmarks_sequence):
    """
    Normalize coordinates relative to torso size.
    """
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    
    # Calculate median torso size across all frames
    torso_sizes = []
    for frame in landmarks_sequence:
        if frame['pose'] is not None:
            pose = frame['pose']
            left_shoulder = pose[LEFT_SHOULDER, :3]
            right_shoulder = pose[RIGHT_SHOULDER, :3]
            left_hip = pose[LEFT_HIP, :3]
            right_hip = pose[RIGHT_HIP, :3]
            
            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
            torso_height = np.linalg.norm((left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2)
            torso_size = (shoulder_width + torso_height) / 2
            
            if torso_size > 1e-6:
                torso_sizes.append(torso_size)
    
    if len(torso_sizes) == 0:
        return None
    
    median_torso_size = np.median(torso_sizes)
    
    # Normalize all frames
    normalized_sequence = []
    for frame in landmarks_sequence:
        normalized_frame = {
            'pose': None,
            'left_hand': None,
            'right_hand': None
        }
        
        if frame['pose'] is not None:
            pose = frame['pose'].copy()
            pose[:, :3] = pose[:, :3] / median_torso_size
            normalized_frame['pose'] = pose
        
        if frame['left_hand'] is not None:
            normalized_frame['left_hand'] = frame['left_hand'].copy() / median_torso_size
        
        if frame['right_hand'] is not None:
            normalized_frame['right_hand'] = frame['right_hand'].copy() / median_torso_size
        
        normalized_sequence.append(normalized_frame)
    
    return normalized_sequence


# ============================================
# APPROACH 1: Temporal Resampling (Fixed Length)
# ============================================

def resample_sequence_to_fixed_length(landmarks_sequence, target_frames=30):
    """
    Resample variable-length sequence to fixed length using interpolation.
    
    Args:
        landmarks_sequence: list of frame dicts
        target_frames: desired number of frames (default 30 = ~1 second at 30fps)
    
    Returns:
        Fixed-length flattened feature vector
    """
    # Normalize first
    landmarks_sequence = normalize_landmarks(landmarks_sequence)
    if landmarks_sequence is None:
        return None
    
    # Extract pose landmarks only (33 landmarks × 3 coords = 99 features per frame)
    pose_frames = []
    for frame in landmarks_sequence:
        if frame['pose'] is not None:
            # Use only x, y, z (drop visibility)
            pose_frames.append(frame['pose'][:, :3].flatten())
    
    if len(pose_frames) < 2:
        return None
    
    pose_frames = np.array(pose_frames)  # shape: (original_frames, 99)
    original_frames = len(pose_frames)
    
    # Interpolate to target length
    from scipy.interpolate import interp1d
    
    original_time = np.linspace(0, 1, original_frames)
    target_time = np.linspace(0, 1, target_frames)
    
    resampled_frames = []
    for dim in range(pose_frames.shape[1]):  # For each of 99 dimensions
        interpolator = interp1d(original_time, pose_frames[:, dim], kind='linear')
        resampled_frames.append(interpolator(target_time))
    
    resampled_frames = np.array(resampled_frames).T  # shape: (target_frames, 99)
    
    # Flatten to single vector: target_frames × 99 features
    feature_vector = resampled_frames.flatten()
    
    return feature_vector


# ============================================
# APPROACH 2: Statistical Pooling (Variable Length)
# ============================================

def extract_statistical_features_from_landmarks(landmarks_sequence):
    """
    Extract statistical summaries from each landmark across time.
    Works with variable-length sequences.
    
    For each landmark and each coordinate (x, y, z):
    - mean, std, min, max across time
    
    Returns fixed-length vector regardless of sequence length.
    """
    # Normalize first
    landmarks_sequence = normalize_landmarks(landmarks_sequence)
    if landmarks_sequence is None:
        return None
    
    # Extract pose landmarks
    pose_frames = []
    for frame in landmarks_sequence:
        if frame['pose'] is not None:
            pose_frames.append(frame['pose'][:, :3])  # shape: (33, 3)
    
    if len(pose_frames) < 2:
        return None
    
    pose_frames = np.array(pose_frames)  # shape: (frames, 33, 3)
    
    features = []
    
    # For each landmark (33 total)
    for landmark_idx in range(33):
        # For each coordinate (x, y, z)
        for coord_idx in range(3):
            trajectory = pose_frames[:, landmark_idx, coord_idx]
            
            # Statistical features
            features.extend([
                np.mean(trajectory),
                np.std(trajectory),
                np.min(trajectory),
                np.max(trajectory),
            ])
    
    # Total: 33 landmarks × 3 coords × 4 stats = 396 features
    return np.array(features)


# ============================================
# APPROACH 3: Dynamic Time Warping (DTW) Kernel
# ============================================

def compute_dtw_distance(seq1, seq2):
    """
    Compute Dynamic Time Warping distance between two sequences.
    Handles variable-length sequences naturally.
    """
    from scipy.spatial.distance import euclidean
    
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.zeros((n + 1, m + 1))
    
    # Initialize
    for i in range(n + 1):
        dtw_matrix[i, 0] = np.inf
    for j in range(m + 1):
        dtw_matrix[0, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    # Fill matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean(seq1[i-1], seq2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # insertion
                dtw_matrix[i, j-1],    # deletion
                dtw_matrix[i-1, j-1]   # match
            )
    
    return dtw_matrix[n, m]


def create_dtw_kernel_matrix(X_sequences):
    """
    Create a kernel matrix using DTW distances.
    This allows SVM to work with variable-length sequences directly.
    """
    n_samples = len(X_sequences)
    kernel_matrix = np.zeros((n_samples, n_samples))
    
    print("Computing DTW kernel matrix...")
    for i in range(n_samples):
        if i % 10 == 0:
            print(f"  Progress: {i}/{n_samples}")
        for j in range(i, n_samples):
            if X_sequences[i] is None or X_sequences[j] is None:
                kernel_matrix[i, j] = np.inf
                kernel_matrix[j, i] = np.inf
            else:
                dist = compute_dtw_distance(X_sequences[i], X_sequences[j])
                # Convert distance to similarity (RBF-like kernel)
                gamma = 0.1
                similarity = np.exp(-gamma * dist)
                kernel_matrix[i, j] = similarity
                kernel_matrix[j, i] = similarity
    
    return kernel_matrix


def prepare_sequences_for_dtw(landmarks_sequence):
    """
    Extract pose landmarks as sequence for DTW.
    """
    landmarks_sequence = normalize_landmarks(landmarks_sequence)
    if landmarks_sequence is None:
        return None
    
    pose_frames = []
    for frame in landmarks_sequence:
        if frame['pose'] is not None:
            pose_frames.append(frame['pose'][:, :3].flatten())  # Flatten to 99-dim vector
    
    if len(pose_frames) < 2:
        return None
    
    return np.array(pose_frames)


# ============================================
# MAIN SCRIPT
# ============================================

def main():
    data_dir = '/Users/alisonyu/asl_2nd_pkl'  # Change to your directory
    
    print("Loading data...")
    X_raw, y, filenames = load_holistic_landmarks(data_dir)
    print(f"Loaded {len(X_raw)} videos")
    print(f"Class distribution: {np.unique(y, return_counts=True)}")
    
    # ============================================
    # APPROACH 1: Temporal Resampling
    # ============================================
    print("\n" + "="*60)
    print("APPROACH 1: Temporal Resampling (Fixed 30 frames)")
    print("="*60)
    
    X_resampled = []
    valid_indices_1 = []
    for i, seq in enumerate(X_raw):
        features = resample_sequence_to_fixed_length(seq, target_frames=30)
        if features is not None and np.isfinite(features).all():
            X_resampled.append(features)
            valid_indices_1.append(i)
    
    X_resampled = np.array(X_resampled)
    y_resampled = y[valid_indices_1]
    filenames_resampled = [filenames[i] for i in valid_indices_1]
    
    print(f"Valid samples: {len(X_resampled)}/{len(X_raw)}")
    print(f"Feature dimensions: {X_resampled.shape[1]} (30 frames × 99 coords)")
    
    # Train SVM
    pipeline_1 = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', C=1.0, random_state=42))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_1 = cross_val_score(pipeline_1, X_resampled, y_resampled, cv=cv, scoring='accuracy')
    
    print(f"\nResults:")
    print(f"Cross-validation accuracy: {scores_1.mean():.3f} (+/- {scores_1.std():.3f})")
    print(f"Individual fold scores: {scores_1}")
    
    # ============================================
    # APPROACH 2: Statistical Pooling
    # ============================================
    print("\n" + "="*60)
    print("APPROACH 2: Statistical Pooling")
    print("="*60)
    
    X_statistical = []
    valid_indices_2 = []
    for i, seq in enumerate(X_raw):
        features = extract_statistical_features_from_landmarks(seq)
        if features is not None and np.isfinite(features).all():
            X_statistical.append(features)
            valid_indices_2.append(i)
    
    X_statistical = np.array(X_statistical)
    y_statistical = y[valid_indices_2]
    filenames_statistical = [filenames[i] for i in valid_indices_2]
    
    print(f"Valid samples: {len(X_statistical)}/{len(X_raw)}")
    print(f"Feature dimensions: {X_statistical.shape[1]} (33 landmarks × 3 coords × 4 stats)")
    
    # Train SVM
    pipeline_2 = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', C=1.0, random_state=42))
    ])
    
    scores_2 = cross_val_score(pipeline_2, X_statistical, y_statistical, cv=cv, scoring='accuracy')
    
    print(f"\nResults:")
    print(f"Cross-validation accuracy: {scores_2.mean():.3f} (+/- {scores_2.std():.3f})")
    print(f"Individual fold scores: {scores_2}")
    
    # ============================================
    # APPROACH 3: DTW Kernel (Advanced, slower)
    # ============================================
    print("\n" + "="*60)
    print("APPROACH 3: Dynamic Time Warping Kernel")
    print("="*60)
    print("Warning: This approach is computationally expensive!")
    
    use_dtw = input("Run DTW approach? (y/n): ").lower() == 'y'
    
    if use_dtw:
        X_sequences = []
        valid_indices_3 = []
        for i, seq in enumerate(X_raw):
            seq_array = prepare_sequences_for_dtw(seq)
            if seq_array is not None:
                X_sequences.append(seq_array)
                valid_indices_3.append(i)
            else:
                X_sequences.append(None)
        
        y_dtw = y[valid_indices_3]
        
        # Compute kernel matrix
        K = create_dtw_kernel_matrix([X_sequences[i] for i in valid_indices_3])
        
        # SVM with precomputed kernel
        svm_dtw = SVC(kernel='precomputed', C=1.0, random_state=42)
        scores_3 = cross_val_score(svm_dtw, K, y_dtw, cv=cv, scoring='accuracy')
        
        print(f"\nResults:")
        print(f"Cross-validation accuracy: {scores_3.mean():.3f} (+/- {scores_3.std():.3f})")
        print(f"Individual fold scores: {scores_3}")
    
    # ============================================
    # COMPARISON SUMMARY
    # ============================================
    print("\n" + "="*60)
    print("SUMMARY: Comparison of Approaches")
    print("="*60)
    print(f"Approach 1 (Resampling):        {scores_1.mean():.3f} (+/- {scores_1.std():.3f})")
    print(f"Approach 2 (Statistical):       {scores_2.mean():.3f} (+/- {scores_2.std():.3f})")
    if use_dtw:
        print(f"Approach 3 (DTW Kernel):        {scores_3.mean():.3f} (+/- {scores_3.std():.3f})")
    print(f"\nFor reference:")
    print(f"Handcrafted features (baseline): 0.533 (+/- 0.027)")
    
    # Analyze misclassified samples
    print("\n" + "="*60)
    print("MISCLASSIFICATION ANALYSIS (Approach 2)")
    print("="*60)
    
    pipeline_2.fit(X_statistical, y_statistical)
    y_pred = cross_val_predict(pipeline_2, X_statistical, y_statistical, cv=cv)
    
    errors = []
    for i, (true, pred) in enumerate(zip(y_statistical, y_pred)):
        if true != pred:
            errors.append((filenames_statistical[i], true, pred))
    
    print(f"Total errors: {len(errors)}/{len(y_statistical)}")
    print("\nMisclassified samples:")
    for fname, true, pred in errors:
        print(f"  {fname}: labeled {true}, predicted {pred}")


if __name__ == "__main__":
    main()
