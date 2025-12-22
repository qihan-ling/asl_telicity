# This script is generated with the asistance of AI
import itertools
from sklearn.inspection import permutation_importance
from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from scipy.stats import chi2, mannwhitneyu
from statsmodels.stats.multitest import multipletests


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
                raise ValueError(
                    f"File {file} does not contain telic/atelic label")

    return X, np.array(y), filenames


def normalize_landmarks(landmarks_sequence):
    """
    Normalize coordinates relative to torso size to make measurements scale-invariant.


    landmarks_sequence: list of dicts with keys 'pose', 'left_hand', 'right_hand'
    Returns: normalized landmarks_sequence
    """
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    normalized_sequence = []

    for frame in landmarks_sequence:
        normalized_frame = {
            'pose': None,
            'left_hand': None,
            'right_hand': None
        }

        if frame['pose'] is not None:
            pose = frame['pose'].copy()

            # Calculate torso size (shoulder width + torso height averaged)
            left_shoulder = pose[LEFT_SHOULDER, :3]
            right_shoulder = pose[RIGHT_SHOULDER, :3]
            left_hip = pose[LEFT_HIP, :3]
            right_hip = pose[RIGHT_HIP, :3]

            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
            torso_height = np.linalg.norm(
                (left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2)
            torso_size = (shoulder_width + torso_height) / 2

            # Normalize pose coordinates
            if torso_size > 1e-6:  # Avoid division by zero
                pose[:, :3] = pose[:, :3] / torso_size

            normalized_frame['pose'] = pose

        # Hands are already in relative coordinates (0-1 range) from MediaPipe
        # But we can normalize them relative to torso for consistency
        if frame['left_hand'] is not None and frame['pose'] is not None:
            normalized_frame['left_hand'] = frame['left_hand'].copy()
            if torso_size > 1e-6:
                normalized_frame['left_hand'] = normalized_frame['left_hand'] / torso_size

        if frame['right_hand'] is not None and frame['pose'] is not None:
            normalized_frame['right_hand'] = frame['right_hand'].copy()
            if torso_size > 1e-6:
                normalized_frame['right_hand'] = normalized_frame['right_hand'] / torso_size

        normalized_sequence.append(normalized_frame)

    return normalized_sequence


def compute_phonological_features(landmarks_sequence):
    """
    Compute phonological features from MediaPipe Holistic landmarks.
    landmarks_sequence: list of dicts with keys 'pose', 'left_hand', 'right_hand'


    Returns: numpy array of phonological features
    """
    # First normalize the landmarks
    landmarks_sequence = normalize_landmarks(landmarks_sequence)

    # MediaPipe Hand landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    # MediaPipe Pose landmark indices
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    RIGHT_WRIST = 16
    LEFT_WRIST = 15

    # Determine dominant hand (which appears more often)
    right_hand_frames = sum(
        1 for f in landmarks_sequence if f['right_hand'] is not None)
    left_hand_frames = sum(
        1 for f in landmarks_sequence if f['left_hand'] is not None)

    if right_hand_frames == 0 and left_hand_frames == 0:
        return np.zeros(15)  # No hand detected (updated feature count)

    hand_key = 'right_hand' if right_hand_frames >= left_hand_frames else 'left_hand'
    wrist_idx = RIGHT_WRIST if hand_key == 'right_hand' else LEFT_WRIST

    # Filter frames where hand is detected
    hand_frames = [f[hand_key]
                   for f in landmarks_sequence if f[hand_key] is not None]
    pose_frames = [f['pose']
                   for f in landmarks_sequence if f['pose'] is not None]

    if len(hand_frames) < 2 or len(pose_frames) < 2:
        return np.zeros(15)  # Not enough data

    # Convert to numpy arrays
    hand_frames = np.array(hand_frames)
    pose_frames = np.array(pose_frames)
    # --- 1. ORIENTATION CHANGE (Extension, Flexion, Pronation, Supination) ---

    # Landmarks for wrist and elbow
    WRIST = 0
    ELBOW = 6   # For right hand: 14, adjust if using pose not hand
    # If using holistic pose: LEFT_ELBOW = 13, RIGHT_ELBOW = 14
    # Use correct one depending on left/right hand detection:
    ELBOW = 14 if hand_key == 'right_hand' else 13

    # Hand base joints
    INDEX_BASE = 5   # MCP joint
    PINKY_BASE = 17

    # Extract orientation-relevant vectors for each hand frame
    palm_normals = []
    forearm_dirs = []   # from elbow to wrist

    # We need corresponding pose frames to extract elbow location
    pose_for_hand = [
        f['pose'] for f in landmarks_sequence if f['pose'] is not None and f[hand_key] is not None
    ]

    hand_frames_sync = [
        f[hand_key] for f in landmarks_sequence if f['pose'] is not None and f[hand_key] is not None
    ]

    def safe_unit(v):
        n = np.linalg.norm(v)
        return v / (n + 1e-8)

    def compute_palm_normal(hand):
        """Return the unit normal vector of the palm plane."""
        v1 = hand[INDEX_BASE] - hand[WRIST]
        v2 = hand[PINKY_BASE] - hand[WRIST]
        normal = np.cross(v1, v2)
        return safe_unit(normal)

    for hand, pose in zip(hand_frames_sync, pose_for_hand):
        wrist_pos = pose[WRIST, :3]
        elbow_pos = pose[ELBOW, :3]

        forearm_vec = wrist_pos - elbow_pos
        forearm_dir = safe_unit(forearm_vec)

        palm_normal = compute_palm_normal(hand)

        palm_normals.append(palm_normal)
        forearm_dirs.append(forearm_dir)

    palm_normals = np.array(palm_normals)
    forearm_dirs = np.array(forearm_dirs)

    def angle_between(v1, v2):
        """Unsigned angle between two vectors in degrees."""
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    def project_onto_plane(v, normal):
        """Remove the component of v along normal."""
        return v - np.dot(v, normal) * normal

    def signed_rotation_around_axis(N, N0, F):
        """
        Signed pronation/supination:
        rotation of palm normal N around forearm axis F,
        relative to reference normal N0.
        """
        # project onto plane perpendicular to F
        Np = project_onto_plane(N,  F)
        N0p = project_onto_plane(N0, F)

        # normalize projections
        Np /= np.linalg.norm(Np)
        N0p /= np.linalg.norm(N0p)

        # unsigned rotation magnitude
        ang = angle_between(Np, N0p)

        # determine sign (direction of twist)
        sign = np.sign(np.dot(F, np.cross(N0p, Np)))
        return ang * sign

    def extract_orientation_features(palm_normals, forearm_dirs):
        # -------------------------------------------------------------
        # 1. Establish reference palm normal (for pronation/supination)
        #    The reference is defined relative to the starting forearm.
        #    We project first palm normal into plane perpendicular to forearm,
        #    so twist = 0 at frame 0.
        # -------------------------------------------------------------
        f0 = forearm_dirs[0]
        pn0 = palm_normals[0]

        # Remove forearm component to get perpendicular palm normal direction
        pn0_proj = pn0 - np.dot(pn0, f0) * f0
        if np.linalg.norm(pn0_proj) < 1e-6:
            # fallback if projection degenerates
            pn0_proj = np.array([1.0, 0.0, 0.0])

        ref_palm_normal = pn0_proj / np.linalg.norm(pn0_proj)

        angle_values = []
        rotation_values = []

        for N, F in zip(palm_normals, forearm_dirs):

            # ----- flexion/extension angle (unsigned) -----
            angle = angle_between(N, F)
            angle_values.append(angle)

            # ----- pronation/supination twist (signed) -----
            rotation = signed_rotation_around_axis(N, ref_palm_normal, F)
            # "either direction" → use magnitude
            rotation_values.append(abs(rotation))

        # compute the four required features
        max_angle = np.max(angle_values)
        sum_angle = np.sum(angle_values)

        max_rotation = np.max(rotation_values)
        sum_rotation = np.sum(rotation_values)

        return max_angle, sum_angle, max_rotation, sum_rotation

    max_angle, sum_angle, max_rotation, sum_rotation = extract_orientation_features(
        palm_normals, forearm_dirs)

    # --- 2. APERTURE CHANGE ---
    # Mean pairwise distance between fingertips
    fingertips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

    def get_aperture(hand):
        distances = []
        for i in range(len(fingertips)):
            for j in range(i+1, len(fingertips)):
                dist = np.linalg.norm(
                    hand[fingertips[i]] - hand[fingertips[j]])
                distances.append(dist)
        return np.mean(distances)

    apertures = np.array([get_aperture(hand) for hand in hand_frames])
    aperture_change = abs(apertures[-1] - apertures[0])
    aperture_range = np.max(apertures) - np.min(apertures)
    aperture_std = np.std(apertures)

    # --- 3. Setting CHANGE ---

    # Regions you want to track
    FOCUS_REGIONS = ["above_head", "head", "torso", "below_torso"]
    DIMENSIONS = ["side", "vertical", "depth"]

    nose_y = np.mean(pose_frames[:, NOSE, 1])
    shoulder_y = np.mean(
        pose_frames[:, LEFT_SHOULDER, 1] + pose_frames[:, RIGHT_SHOULDER, 1] / 2)
    hip_y = np.mean((pose_frames[:, LEFT_HIP, 1] +
                     pose_frames[:, RIGHT_HIP, 1]) / 2)

    def compute_signed_change(positions):
        """
        Compute signed movement change along each axis for one body region.
        Movements in opposite directions cancel out.
        positions: (T×3) XYZ coordinates over time.
        Returns a dict mapping dimension → signed sum.
        """
        if positions.size <= 0:
            return None
        # print(positions.shape)
        xs = positions[:, 0]   # left ↔ right
        ys = positions[:, 1]   # down ↔ up
        zs = positions[:, 2]   # toward ↔ away from body

        dx = np.diff(xs)
        dy = np.diff(ys)
        dz = np.diff(zs)

        # Sum, not abs sum — signed sum allows cancellation
        return {
            "side":     np.sum(dx),
            "vertical": np.sum(dy),
            "depth":    np.sum(dz),
        }

    def extract_setting_changes(regions):
        """
        INPUT:
           regions: dict mapping region_name → (T×3 array)
                    Only the regions in FOCUS_REGIONS are considered.

        OUTPUT:
           np.array of shape (8,) with layout:
                 [4 one-hot for regions,
                 3 one-hot for dimensions,
                 1 signed numeric change]

           The chosen region is the one with the largest |signed change|
           across all regions and dimensions.
        """

        best_region = None
        best_dim = None
        best_signed_value = 0.0
        best_strength = -np.inf  # strength = abs(signed change)

        for region_name in FOCUS_REGIONS:
            pos = regions.get(region_name)
            if pos is None:
                continue

            signed_vals = compute_signed_change(pos)
            if signed_vals is None:
                continue
            else:
                for dim in DIMENSIONS:
                    signed_val = signed_vals[dim]
                    strength = abs(signed_val)

                    if strength > best_strength:
                        best_strength = strength
                        best_signed_value = signed_val
                        best_region = region_name
                        best_dim = dim

        # Create final 8-dim feature vector
        # vec = np.zeros(8, dtype=float)

        if best_region is not None:
            # vec[FOCUS_REGIONS.index(best_region)] = 1.0
            best_region = FOCUS_REGIONS.index(best_region)

        if best_dim is not None:
            # vec[4 + DIMENSIONS.index(best_dim)] = 1.0
            best_dim = DIMENSIONS.index(best_dim)

        # final slot = signed change value (positive or negative)
        # vec[7] = best_signed_value

        return best_region, best_dim, best_signed_value

    def classify_region(wrist_y):
        # Thresholds adjusted for normalized coordinates
        if wrist_y < nose_y - 0.1:  # above head
            return "above_head"
        elif wrist_y < shoulder_y:  # head region
            return "head"
        elif wrist_y < hip_y:  # torso
            return "torso"
        else:  # lower region
            return "below_torso"

    regions = {'above_head': [], 'head': [], 'torso': [], 'below_torso': []}
    for frame in pose_frames:
        wrist = frame[wrist_idx, :3]
        wrist_y = wrist[1]
        region = classify_region(wrist_y)
        # TODO: how to account for the gap of moving from one region to another
        regions[region].append(wrist)
        # Note: for regions: some keys may have empty list values
    for region in regions:
        regions[region] = np.asarray(regions[region])
    best_region, best_dim, best_signed_value = extract_setting_changes(regions)

    # --- 4. CONTACT DETECTION ---
    # A. Hand-to-body contact
    body_landmarks = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    # Threshold adjusted for normalized coordinates (relative to torso size)
    body_contact_threshold = 0.25

    body_contact_frames = 0
    min_body_distance = float('inf')
    body_contact_binary = 0

    for frame in pose_frames:
        wrist_pos = frame[wrist_idx, :3]  # x, y, z
        for body_idx in body_landmarks:
            body_pos = frame[body_idx, :3]
            distance = np.linalg.norm(wrist_pos - body_pos)
            min_body_distance = min(min_body_distance, distance)

            if distance < body_contact_threshold:
                body_contact_frames += 1
                body_contact_binary = 1

    # B. Hand-to-hand contact
    hand_contact_threshold = 0.20  # adjusted for normalized coordinates
    hand_contact_frames = 0
    min_hand_distance = float('inf')
    hand_contact_binary = 0

    # Get frames where both hands are detected
    both_hands_frames = [
        (f['left_hand'], f['right_hand'])
        for f in landmarks_sequence
        if f['left_hand'] is not None and f['right_hand'] is not None
    ]

    if len(both_hands_frames) > 0:
        for left_hand, right_hand in both_hands_frames:
            frame_min_dist = float('inf')
            for left_lm in left_hand:
                for right_lm in right_hand:
                    distance = np.linalg.norm(left_lm - right_lm)
                    if np.isfinite(distance):
                        frame_min_dist = min(frame_min_dist, distance)
                        min_hand_distance = min(min_hand_distance, distance)

            if frame_min_dist < hand_contact_threshold:
                hand_contact_frames += 1
                hand_contact_binary = 1

    else:
        min_hand_distance = 10.0

    if not np.isfinite(min_hand_distance):
        min_hand_distance = 10.0

    # --- ADDITIONAL SPATIAL FEATURES ---
    # Path length and displacement of hand
    wrist_trajectory = pose_frames[:, wrist_idx, :3]
    path_length = np.sum(np.linalg.norm(
        np.diff(wrist_trajectory, axis=0), axis=1))
    displacement = np.linalg.norm(wrist_trajectory[-1] - wrist_trajectory[0])
    straightness = displacement / (path_length + 1e-8)

    # Compile feature vector
    features = np.array([
        max_angle,
        sum_angle,
        max_rotation,
        sum_rotation,
        aperture_change,
        aperture_range,
        aperture_std,
        best_region,
        best_dim,
        best_signed_value,
        body_contact_frames,
        min_body_distance,
        body_contact_binary,
        hand_contact_frames,
        min_hand_distance,
        hand_contact_binary,
        straightness
    ])

    return features


def compute_motion_features(landmarks_sequence):
    """
    Compute motion features (velocity, acceleration) from pose landmarks.
    landmarks_sequence: list of dicts with keys 'pose', 'left_hand', 'right_hand'


    Returns: numpy array of motion features
    """
    # Normalize first
    landmarks_sequence = normalize_landmarks(landmarks_sequence)

    # Extract pose coordinates across frames
    pose_frames = [f['pose'][:, :3]
                   for f in landmarks_sequence if f['pose'] is not None]

    if len(pose_frames) < 3:
        return np.zeros(9)  # fallback (matches actual feature count)

    pose_frames = np.array(pose_frames)  # shape: (frames, 33, 3)

    # Compute velocities and accelerations
    velocities = np.diff(pose_frames, axis=0)  # shape: (frames-1, 33, 3)
    accelerations = np.diff(velocities, axis=0)  # shape: (frames-2, 33, 3)

    # Flatten to get overall body motion
    velocity_norms = np.linalg.norm(
        velocities.reshape(velocities.shape[0], -1), axis=1)
    acceleration_norms = np.linalg.norm(
        accelerations.reshape(accelerations.shape[0], -1), axis=1)

    features = np.array([
        np.mean(velocity_norms),      # 0: mean velocity
        np.std(velocity_norms),       # 1: velocity variability
        np.max(velocity_norms),       # 2: peak velocity
        np.min(velocity_norms),       # 3: minimum velocity
        np.mean(acceleration_norms),  # 4: mean acceleration
        np.std(acceleration_norms),   # 5: acceleration variability
        np.max(acceleration_norms),   # 6: peak acceleration
        np.min(acceleration_norms),   # 7: minimum acceleration
        np.sum(velocity_norms),       # 8: total path length (velocity sum)
    ])

    return features


def compute_all_features(landmarks_sequence):
    """
    Compute both phonological and motion features.
    Returns: combined feature vector
    """
    phono_features = compute_phonological_features(landmarks_sequence)
    motion_features = compute_motion_features(landmarks_sequence)

    # Combine features
    all_features = np.concatenate([phono_features, motion_features])
    return all_features


def visualize_features(X_features, y, feature_names):
    """
    Visualize feature distributions for telic vs atelic signs.
    """
    n_features = X_features.shape[1]
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()

    for i in range(n_features):
        ax = axes[i]
        telic_values = X_features[y == 'telic', i]
        atelic_values = X_features[y == 'atelic', i]

        ax.hist(telic_values, alpha=0.5, label='telic', bins=10, color='blue')
        ax.hist(atelic_values, alpha=0.5, label='atelic', bins=10, color='red')
        ax.set_title(feature_names[i], fontsize=10)
        ax.legend()
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('new_feature_distributions.png', dpi=150, bbox_inches='tight')
    print("Saved feature distributions to 'feature_distributions.png'")
    plt.close()


def train_and_evaluate_svm(X_features, y):
    """
    Train SVM with cross-validation and report results.
    """
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', C=1.0, random_state=42))
    ])

    # Stratified K-Fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_features, y,
                             cv=cv, scoring='accuracy')

    print("\n=== SVM Classification Results ===")
    print(
        f"Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    print(f"Individual fold scores: {scores}")

    # Train on full dataset to get feature weights
    pipeline.fit(X_features, y)
    with open('telicity_model_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    svm_model = pipeline.named_steps['svm']
    scaler = pipeline.named_steps['scaler']

    # Get feature importance (absolute weights)
    feature_weights = np.abs(svm_model.coef_[0])

    return feature_weights, pipeline


def analyze_feature_importance(feature_weights, feature_names):
    """
    Analyze and display feature importance.
    """
    # Sort features by importance
    indices = np.argsort(feature_weights)[::-1]

    print("\n=== Feature Importance (SVM weights) ===")
    for i, idx in enumerate(indices):
        print(f"{i+1}. {feature_names[idx]}: {feature_weights[idx]:.4f}")

    # Plot feature importance
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(feature_weights)),
             feature_weights[indices], color='steelblue')
    plt.yticks(range(len(feature_weights)), [
               feature_names[i] for i in indices])
    plt.xlabel('Absolute SVM Weight (Feature Importance)')
    plt.title('Feature Importance for Telicity Detection')
    plt.tight_layout()
    plt.savefig('new_feature_importance.png', dpi=150, bbox_inches='tight')
    print("\nSaved feature importance plot to 'feature_importance.png'")
    plt.close()


def diagnose_invalid_features(X, y, filenames):
    """
    Diagnose which samples and features have invalid values.
    """
    print("\n=== Diagnosing Invalid Features ===")

    for idx, (landmarks, label, filename) in enumerate(zip(X, y, filenames)):
        features = compute_all_features(landmarks)

        if not np.isfinite(features).all():
            print(f"\n[{idx}] {filename} ({label})")

            # Check which features are invalid
            invalid_indices = np.where(~np.isfinite(features))[0]
            print(f"  Invalid feature indices: {invalid_indices}")

            # Analyze the raw data
            pose_frames = [f['pose']
                           for f in landmarks if f['pose'] is not None]
            right_hand_frames = [f['right_hand']
                                 for f in landmarks if f['right_hand'] is not None]
            left_hand_frames = [f['left_hand']
                                for f in landmarks if f['left_hand'] is not None]

            print(f"  Total frames: {len(landmarks)}")
            print(f"  Pose frames: {len(pose_frames)}")
            # print(f"  Right hand frames: {right_hand_frames}")
            # print(f"  Left hand frames: {left_hand_frames}")

            # Check torso size
            if len(pose_frames) > 0:
                pose = pose_frames[0]
                LEFT_SHOULDER = 11
                RIGHT_SHOULDER = 12
                LEFT_HIP = 23
                RIGHT_HIP = 24

                left_shoulder = pose[LEFT_SHOULDER, :3]
                right_shoulder = pose[RIGHT_SHOULDER, :3]
                left_hip = pose[LEFT_HIP, :3]
                right_hip = pose[RIGHT_HIP, :3]

                shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
                torso_height = np.linalg.norm(
                    (left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2)
                torso_size = (shoulder_width + torso_height) / 2

                print(f"  Torso size: {torso_size:.6f}")
                print(f"  Shoulder width: {shoulder_width:.6f}")
                print(f"  Torso height: {torso_height:.6f}")


# ===================================================================
# 0. STANDARDIZATION
# ===================================================================

def standardize_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ===================================================================
# 1. LOGISTIC REGRESSION LIKELIHOOD-RATIO TEST
# ===================================================================

def likelihood_ratio_test(X, y):
    """
    Perform likelihood ratio test comparing full model to null (intercept-only) model.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix (should be numeric, scaled)
    y : array-like, shape (n_samples,)
        Binary labels (should be numeric: 0/1)
    
    Returns:
    --------
    p_value : float
    lr_stat : float (likelihood ratio statistic)
    df : int (degrees of freedom = number of features)
    """
    import warnings
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Ensure y is numeric (0/1)
    if y.dtype.kind in ['U', 'S', 'O']:  # string types
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Suppress Hessian inversion warnings - we only need log-likelihood, not standard errors
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=Warning)
        
        # null model: intercept only
        X_null = sm.add_constant(np.ones(len(y)))
        try:
            model_null = sm.Logit(y, X_null).fit(disp=False, method='bfgs', maxiter=1000)
        except Exception:
            try:
                model_null = sm.Logit(y, X_null).fit(disp=False, method='nm', maxiter=1000)
            except Exception:
                model_null = sm.Logit(y, X_null).fit(disp=False)

        # full model with regularization to improve numerical stability
        X_full = sm.add_constant(X)
        try:
            # Try L2 regularization (Ridge) for stability
            model_full = sm.Logit(y, X_full).fit_regularized(
                disp=False, method='l1', alpha=0.1, maxiter=1000
            )
        except Exception:
            try:
                model_full = sm.Logit(y, X_full).fit(disp=False, method='bfgs', maxiter=1000)
            except Exception:
                try:
                    model_full = sm.Logit(y, X_full).fit(disp=False, method='nm', maxiter=1000)
                except Exception:
                    model_full = sm.Logit(y, X_full).fit(disp=False)

    lr_stat = 2 * (model_full.llf - model_null.llf)
    # Ensure LR stat is non-negative (can be slightly negative due to numerical issues)
    lr_stat = max(0, lr_stat)
    
    df = X.shape[1]  # degrees of freedom = number of features being tested
    p_value = chi2.sf(lr_stat, df)  # survival function (1 - CDF)

    return p_value, lr_stat, df


# ===================================================================
# 2. UNIVARIATE FEATURE TESTS
# ===================================================================

def test_single_features(X_scaled, y, feature_names=None):
    output = []

    for i in range(X_scaled.shape[1]):
        p, lr, df = likelihood_ratio_test(X_scaled[:, [i]], y)
        fname = feature_names[i] if feature_names else f"F{i}"
        output.append({"feature": fname, "p": p, "LR": lr})

    return pd.DataFrame(output).sort_values("p")


# ===================================================================
# 3. MULTIVARIATE FEATURE GROUP TESTS
# ===================================================================

def test_feature_group(X_scaled, y, group_indices, feature_names=None):
    group_X = X_scaled[:, group_indices]
    p, lr, df = likelihood_ratio_test(group_X, y)

    if feature_names:
        label = "+".join([feature_names[i] for i in group_indices])
    else:
        label = str(group_indices)

    return pd.DataFrame([{
        "feature_group": label,
        "p": p,
        "LR": lr,
        "df": df
    }])


# ===================================================================
# 4. PAIRWISE INTERACTION TESTS
# ===================================================================

def test_feature_interaction(X_scaled, y, idx1, idx2, feature_names=None):
    x1 = X_scaled[:, idx1]
    x2 = X_scaled[:, idx2]
    interaction = (x1 * x2).reshape(-1, 1)

    X_int = np.column_stack([x1, x2, interaction])

    p, lr, df = likelihood_ratio_test(X_int, y)

    if feature_names:
        label = f"{feature_names[idx1]}_x_{feature_names[idx2]}"
    else:
        label = f"F{idx1}_x_F{idx2}"

    return pd.DataFrame([{
        "interaction": label,
        "p": p,
        "LR": lr,
        "df": df
    }])


def test_all_interactions(X_scaled, y, feature_names=None):
    results = []
    d = X_scaled.shape[1]

    for i, j in itertools.combinations(range(d), 2):
        df = test_feature_interaction(X_scaled, y, i, j, feature_names)
        results.append(df)

    return pd.concat(results).sort_values("p")


# ===================================================================
# 7. MASTER RUNNER
# ===================================================================

def run_full_statistical_pipeline(X, y, feature_groups, feature_names=None, model=None, alpha=0.05):
    """
    Run comprehensive statistical analysis on features.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix (extracted features, NOT raw landmarks)
    y : array-like
        Labels (can be string or numeric)
    feature_groups : dict
        Mapping group name → list of feature indices
    feature_names : list, optional
        Names for each feature
    model : sklearn model or Pipeline, optional
        Trained model for permutation importance. If it's a Pipeline with a scaler,
        raw X will be used; otherwise X_scaled will be used.
    alpha : float
        Significance level for multiple testing correction
    
    Returns:
    --------
    dict with analysis results
    """
    X = np.asarray(X)
    X_raw = X.copy()  # Keep a copy of raw features for permutation importance
    y = np.asarray(y)
    
    # Encode labels if they're strings
    label_encoder = None
    if y.dtype.kind in ['U', 'S', 'O']:  # string types
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        y_encoded = y

    # 1) Standardize for statistical tests
    X_scaled, scaler = standardize_features(X)

    # 2) Single features with multiple testing correction
    print("  Testing single features...")
    single_feature_results = test_single_features(X_scaled, y_encoded, feature_names)
    
    # Apply FDR correction to single feature tests
    if len(single_feature_results) > 0:
        _, p_corrected, _, _ = multipletests(
            single_feature_results['p'].values, 
            alpha=alpha, 
            method='fdr_bh'
        )
        single_feature_results['p_corrected'] = p_corrected
        single_feature_results['significant'] = p_corrected < alpha

    # 3) Feature groups
    print(f"  Testing {len(feature_groups)} feature groups...")
    group_results = []
    for group_name, indices in feature_groups.items():
        df = test_feature_group(X_scaled, y_encoded, indices, feature_names)
        df["group_name"] = group_name
        group_results.append(df)
    
    if group_results:
        group_results = pd.concat(group_results).sort_values("p")
        # Apply FDR correction
        _, p_corrected, _, _ = multipletests(
            group_results['p'].values,
            alpha=alpha,
            method='fdr_bh'
        )
        group_results['p_corrected'] = p_corrected
        group_results['significant'] = p_corrected < alpha
    else:
        group_results = pd.DataFrame()

    # 4) Interactions with FDR correction
    print("  Testing pairwise interactions...")
    interaction_results = test_all_interactions(X_scaled, y_encoded, feature_names)
    
    if len(interaction_results) > 0:
        _, p_corrected, _, _ = multipletests(
            interaction_results['p'].values,
            alpha=alpha,
            method='fdr_bh'
        )
        interaction_results['p_corrected'] = p_corrected
        interaction_results['significant'] = p_corrected < alpha

    # 5) Permutation importance (if model provided)
    perm_importance_results = None
    if model is not None:
        print("  Computing permutation importance...")
        # IMPORTANT: Use original y labels (not encoded) if model was trained on strings
        # The pipeline was trained with string labels ('telic', 'atelic')
        y_for_perm = y  # Use original labels to match what model was trained on
        
        # Check if model is a Pipeline with a scaler - if so, use raw X
        # (the pipeline will handle scaling internally)
        if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
            # Pipeline with scaler - use raw features
            perm_importance_results = compute_permutation_importance(
                model, X_raw, y_for_perm, feature_names
            )
        else:
            # Just a model without scaler - use scaled features
            perm_importance_results = compute_permutation_importance(
                model, X_scaled, y_for_perm, feature_names
            )

    return {
        "scaled_features": X_scaled,
        "single_feature_results": single_feature_results,
        "group_results": group_results,
        "interaction_results": interaction_results,
        "permutation_importance": perm_importance_results,
        "scaler": scaler,
        "label_encoder": label_encoder
    }


def diagnose_svm_vs_stats_discrepancy(X, y, feature_names, svm_weights, stat_results):
    """
    Diagnose why some features have high SVM weights but low statistical significance.
    
    This can happen due to:
    1. Suppressor effects (feature removes noise from other predictors)
    2. Multicollinearity (feature is redundant with others)
    3. Non-linear relationships (SVM can capture, univariate tests can't)
    4. Interaction effects (feature only matters in combination)
    """
    import warnings
    
    X = np.asarray(X)
    y_encoded = LabelEncoder().fit_transform(y) if y.dtype.kind in ['U', 'S', 'O'] else y
    
    # Get ALL features sorted by SVM weight
    all_indices = np.argsort(svm_weights)[::-1]
    
    # Top 5 features used as baseline for conditional tests
    top5_indices = all_indices[:5].tolist()
    
    print("\n" + "="*60)
    print("DIAGNOSTIC: SVM Weight vs Statistical Significance (ALL FEATURES)")
    print("="*60)
    print(f"Baseline features for conditional test: {[feature_names[i] for i in top5_indices]}")
    
    results = []
    
    for idx in all_indices:
        fname = feature_names[idx]
        svm_weight = svm_weights[idx]
        
        # Get univariate p-value from stat_results
        univar_row = stat_results[stat_results['feature'] == fname]
        univar_p = univar_row['p_corrected'].values[0] if len(univar_row) > 0 else np.nan
        
        # Check correlation with top 5 features
        correlations = []
        for other_idx in top5_indices:
            if other_idx != idx:
                corr = np.corrcoef(X[:, idx], X[:, other_idx])[0, 1]
                if abs(corr) > 0.3:  # moderate correlation
                    correlations.append((feature_names[other_idx], corr))
        
        # Check if it's a suppressor: test if adding it improves a model with top features
        # Use top 3 features (excluding current feature) as baseline
        other_top3 = [i for i in top5_indices if i != idx][:3]
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            # Model with other features only
            X_others = sm.add_constant(X[:, other_top3])
            try:
                model_others = sm.Logit(y_encoded, X_others).fit(disp=False, maxiter=1000)
                llf_others = model_others.llf
            except:
                llf_others = np.nan
            
            # Model with other features + this feature
            X_with_this = sm.add_constant(X[:, other_top3 + [idx]])
            try:
                model_with = sm.Logit(y_encoded, X_with_this).fit(disp=False, maxiter=1000)
                llf_with = model_with.llf
            except:
                llf_with = np.nan
        
        # Likelihood ratio improvement
        lr_improvement = 2 * (llf_with - llf_others) if np.isfinite(llf_with) and np.isfinite(llf_others) else np.nan
        conditional_p = chi2.sf(lr_improvement, 1) if np.isfinite(lr_improvement) and lr_improvement > 0 else np.nan
        
        results.append({
            'feature': fname,
            'svm_weight': svm_weight,
            'univar_p': univar_p,
            'conditional_p': conditional_p,
            'lr_improvement': lr_improvement,
            'correlated_with': correlations[:3] if correlations else 'None'
        })
        
        # Print diagnosis
        is_significant = univar_p < 0.05 if np.isfinite(univar_p) else False
        cond_significant = conditional_p < 0.05 if np.isfinite(conditional_p) else False
        
        # Classification:
        # - Univariate: significant on its own (direct predictor)
        # - Suppressor: not significant alone, but significant when controlling for other features
        # - Interaction: only matters in specific feature combinations (neither test catches it)
        status = "✓ Univariate" if is_significant else ("✓ Suppressor" if cond_significant else "? Interaction")
        
        print(f"\n{fname}:")
        print(f"  SVM weight: {svm_weight:.4f}")
        print(f"  Univariate p: {univar_p:.4f}" + (" ***" if is_significant else ""))
        print(f"  Conditional p (given top features): {conditional_p:.4f}" + (" ***" if cond_significant else "") if np.isfinite(conditional_p) else "  Conditional p: N/A")
        if correlations:
            print(f"  Correlated with: {', '.join([f'{n}({c:.2f})' for n,c in correlations[:3]])}")
        print(f"  Status: {status}")
    
    return pd.DataFrame(results)


def compute_permutation_importance(model, X, y, feature_names=None, n_repeats=30):
    """
    Compute permutation importance for a trained model.
    
    Returns DataFrame with importance scores and p-values.
    """
    result = permutation_importance(
        model, X, y, 
        n_repeats=n_repeats, 
        random_state=42,
        scoring='accuracy'
    )
    
    importance_df = pd.DataFrame({
        'feature': feature_names if feature_names else [f'F{i}' for i in range(X.shape[1])],
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,
    })
    
    # Compute approximate p-values (one-sided test: importance > 0)
    # Using t-test approximation
    from scipy.stats import t as t_dist
    t_stats = result.importances_mean / (result.importances_std / np.sqrt(n_repeats) + 1e-10)
    p_values = t_dist.sf(t_stats, df=n_repeats-1)  # one-sided
    importance_df['p_value'] = p_values
    
    return importance_df.sort_values('importance_mean', ascending=False)


def main():
    """
    Main pipeline for telicity detection.
    """
    # Configuration
    # Change this to your data directory
    data_dir = '/Users/alisonyu/telicity_codes/asl_3nd_pkl'

    print("Loading data...")
    X, y, filenames = load_holistic_landmarks(data_dir)
    print(f"Loaded {len(X)} videos")
    print(f"Class distribution: {np.unique(y, return_counts=True)}")
    diagnose_invalid_features(X, y, filenames)

    print("\nComputing features (with normalization)...")
    X_features = np.array([compute_all_features(landmarks) for landmarks in X])
    print(f"Feature matrix shape: {X_features.shape}")

    # Feature names (26 total: 17 phonological + 9 motion)
    feature_names = [
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

    # Check for invalid features (NaN or Inf)
    invalid_mask = ~np.isfinite(X_features).all(axis=1)
    if invalid_mask.any():
        print(f"\nWarning: {invalid_mask.sum()} samples have invalid features")
        print("Invalid samples:", [filenames[i]
              for i in np.where(invalid_mask)[0]])
        # Remove invalid samples
        X_features = X_features[~invalid_mask]
        y = y[~invalid_mask]
        print(f"Proceeding with {len(X_features)} valid samples")

    # Visualize feature distributions
    # print("\nVisualizing feature distributions...")
    # visualize_features(X_features, y, feature_names)

    # Train and evaluate SVM
    print("\nTraining SVM classifier...")
    feature_weights, pipeline = train_and_evaluate_svm(X_features, y)
    # feature_weights, pipeline = train_and_evaluate_svm(X, y)

    # Analyze feature importance
    analyze_feature_importance(feature_weights, feature_names)

    # Statistical analysis of phonological features
    print("\n=== Statistical Analysis of ALL Features ===")

    # Mann-Whitney U tests for individual features
    print("\n--- Mann-Whitney U Tests (non-parametric) ---")
    mw_results = []
    for i, name in enumerate(feature_names):
        telic_values = X_features[y == 'telic', i]
        atelic_values = X_features[y == 'atelic', i]

        # Mann-Whitney U test (non-parametric)
        statistic, p_value = mannwhitneyu(
            telic_values, atelic_values, alternative='two-sided')
        
        mw_results.append({
            'feature': name,
            'telic_mean': np.mean(telic_values),
            'telic_std': np.std(telic_values),
            'atelic_mean': np.mean(atelic_values),
            'atelic_std': np.std(atelic_values),
            'U_statistic': statistic,
            'p_value': p_value
        })

    mw_df = pd.DataFrame(mw_results).sort_values('p_value')
    
    # Apply FDR correction
    _, p_corrected, _, _ = multipletests(mw_df['p_value'].values, alpha=0.05, method='fdr_bh')
    mw_df['p_corrected'] = p_corrected
    mw_df['significant'] = p_corrected < 0.05
    
    print("\nTop 10 features by Mann-Whitney U test (FDR corrected):")
    print(mw_df.head(10).to_string(index=False))

    # ================================================================
    # AUTOMATED ANALYSIS DRIVER
    # ================================================================

    # ================================================================
    # 1. DEFINE MEANINGFUL FEATURE GROUPS
    #    Instead of all combinations (which is computationally expensive
    #    and statistically problematic), use theoretically meaningful groups
    # ================================================================

    feature_groups = {
        # ===========================================
        # MAIN GROUPS (A, B, C, D, M)
        # ===========================================
        
        # A: Change of Orientation (A1-A4)
        'A_orientation': [0, 1, 2, 3],  # max_angle, sum_angle, max_rotation, sum_rotation
        
        # B: Change of Aperture (B1-B3)
        'B_aperture': [4, 5, 6],  # aperture_change, aperture_range, aperture_std
        
        # C: Setting Change (C1-C4)
        'C_setting': [7, 8, 9, 16],  # best_region, best_dim, best_signed_value, straightness
        
        # D: Location/Contact Change (D1-D6)
        'D_contact': [10, 11, 12, 13, 14, 15],  # body_contact_frames, min_body_distance, body_contact_binary, hand_contact_frames, min_hand_distance, hand_contact_binary
        
        # M: Motion Features (M1-M9)
        'M_motion': [17, 18, 19, 20, 21, 22, 23, 24, 25],  # all motion features
        
        # ===========================================
        # SUB-GROUPS (for detailed analysis)
        # ===========================================
        
        # Body contact only (D1-D3)
        'D_body_contact': [10, 11, 12],
        
        # Hand contact only (D4-D6)
        'D_hand_contact': [13, 14, 15],
        
        # Velocity only (M1-M4)
        'M_velocity': [17, 18, 19, 20],
        
        # Acceleration only (M5-M8)
        'M_acceleration': [21, 22, 23, 24],
        
        # All phonological features (A+B+C+D = indices 0-16)
        'all_phonological_ABCD': list(range(17)),
        
        # Combined groups
        'AB_orientation_aperture': [0, 1, 2, 3, 4, 5, 6],
        'CD_setting_contact': [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        
        # Top features from SVM weights (will be updated dynamically)
        'top5_svm': list(np.argsort(feature_weights)[::-1][:5]),
        'top10_svm': list(np.argsort(feature_weights)[::-1][:10]),
    }

    # ================================================================
    # 2. RUN FULL PIPELINE
    # ================================================================
    print("\nRunning full statistical pipeline...")
    results = run_full_statistical_pipeline(
        X=X_features,  # FIXED: use extracted features, not raw landmarks
        y=y,
        feature_groups=feature_groups,
        feature_names=feature_names,
        model=pipeline  # Pass trained model for permutation importance
    )

    # ================================================================
    # 3. PRINT SUMMARIES
    # ================================================================
    print("\n" + "="*60)
    print("SINGLE FEATURE RESULTS (Likelihood Ratio Tests)")
    print("="*60)
    single_df = results["single_feature_results"]
    print(f"\nSignificant features (FDR corrected p < 0.05):")
    sig_features = single_df[single_df['significant'] == True]
    if len(sig_features) > 0:
        print(sig_features.to_string(index=False))
    else:
        print("  No features significant after FDR correction")
    print(f"\nAll features (sorted by p-value):")
    print(single_df.to_string(index=False))

    print("\n" + "="*60)
    print("FEATURE GROUP RESULTS")
    print("="*60)
    group_df = results["group_results"]
    print(f"\nSignificant groups (FDR corrected p < 0.05):")
    sig_groups = group_df[group_df['significant'] == True]
    if len(sig_groups) > 0:
        print(sig_groups.to_string(index=False))
    else:
        print("  No feature groups significant after FDR correction")
    print(f"\nAll groups:")
    print(group_df.to_string(index=False))
    
    # ==========================================================================
    # GROUPED BAR CHART: Group Significance with Feature Drivers
    # ==========================================================================
    print("\n" + "="*60)
    print("GENERATING GROUP SIGNIFICANCE BAR CHART")
    print("="*60)
    
    # Main groups to visualize
    main_groups = {
        'A: Orientation': 'A_orientation',
        'B: Aperture': 'B_aperture',
        'C: Setting': 'C_setting',
        'D: Contact': 'D_contact',
        'M: Motion': 'M_motion'
    }
    
    # Label mapping
    label_mapping = {
        'max_angle': 'A1', 'sum_angle': 'A2', 'max_rotation': 'A3', 'sum_rotation': 'A4',
        'aperture_change': 'B1', 'aperture_range': 'B2', 'aperture_std': 'B3',
        'best_region': 'C1', 'best_dim': 'C2', 'best_signed_value': 'C3', 'straightness': 'C4',
        'body_contact_frames': 'D1', 'min_body_distance': 'D2', 'body_contact_binary': 'D3',
        'hand_contact_frames': 'D4', 'min_hand_distance': 'D5', 'hand_contact_binary': 'D6',
        'mean_velocity': 'M1', 'velocity_std': 'M2', 'max_velocity': 'M3', 'min_velocity': 'M4',
        'mean_acceleration': 'M5', 'acceleration_std': 'M6', 'max_acceleration': 'M7',
        'min_acceleration': 'M8', 'total_path_length': 'M9'
    }
    
    # Collect data
    group_names = []
    group_lr_values = []
    group_p_values = []
    group_colors = []
    feature_annotations = []  # List of (group_idx, x_pos, label, value)
    
    for group_display, group_key in main_groups.items():
        # Get group statistics
        group_row = group_df[group_df['group_name'] == group_key]
        if len(group_row) == 0:
            continue
        
        group_lr = group_row['LR'].values[0]
        group_p_corrected = group_row['p_corrected'].values[0] if 'p_corrected' in group_row.columns else group_row['p'].values[0]
        is_sig = group_row['significant'].values[0] if 'significant' in group_row.columns else group_p_corrected < 0.05
        
        group_names.append(group_display)
        group_lr_values.append(group_lr)
        group_p_values.append(group_p_corrected)
        
        # Color by significance: red=most significant, orange, yellow, green=least significant
        if group_p_corrected < 0.001:
            group_colors.append('#e74c3c')  # Red (most significant)
        elif group_p_corrected < 0.01:
            group_colors.append('#f39c12')  # Orange
        elif group_p_corrected < 0.05:
            group_colors.append('#f1c40f')  # Yellow
        else:
            group_colors.append('#2ecc71')  # Green (least significant)
        
        # Get top 2-3 features driving this group
        group_indices = feature_groups[group_key]
        feature_contributions = []
        
        for feat_idx in group_indices:
            feat_name = feature_names[feat_idx]
            feat_label = label_mapping.get(feat_name, feat_name)
            
            # Get individual feature statistics
            feat_row = single_df[single_df['feature'] == feat_name]
            if len(feat_row) > 0:
                feat_lr = feat_row['LR'].values[0]
                contribution_pct = (feat_lr / group_lr * 100) if group_lr > 0 else 0
                feature_contributions.append((feat_label, contribution_pct, feat_lr))
            else:
                # Feature not found in single_df - set to 0
                feature_contributions.append((feat_label, 0, 0))
        
        # Sort by contribution and take top 2
        feature_contributions.sort(key=lambda x: x[1], reverse=True)
        top_features = feature_contributions[:2]
        
        # Debug: print feature contributions for this group
        print(f"\n  {group_display} top features:")
        for label, pct, lr in top_features:
            print(f"    {label}: LR={lr:.3f}, Contribution={pct:.1f}%")
        
        # Store annotations (always top 2, even if contribution is small)
        group_idx = len(group_names) - 1
        for i, (label, pct, lr) in enumerate(top_features):
            feature_annotations.append((group_idx, i, label, lr, pct))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Main bars for groups
    x_pos = np.arange(len(group_names))
    group_bar_width = 0.6
    bars = ax.bar(x_pos, group_lr_values, color=group_colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5, width=group_bar_width)
    
    # Add smaller bars for top 2 features - positioned INSIDE the group bar
    feat_bar_width = 0.15
    spacing = 0.03  # Small spacing between feature bars
    
    # Calculate total width needed for 2 feature bars
    total_feat_width = 2 * feat_bar_width + spacing
    # Starting position (left edge of feature bars within group bar)
    feat_start_offset = (group_bar_width - total_feat_width) / 2
    
    for group_idx, feat_idx, label, feat_lr, pct in feature_annotations:
        # Position feature bars inside the group bar
        # Group bar left edge is at x_pos[group_idx] - group_bar_width/2
        # Feature bars start at group_bar_left + feat_start_offset
        group_bar_left = x_pos[group_idx] - group_bar_width / 2
        feat_bar_x = group_bar_left + feat_start_offset + feat_idx * (feat_bar_width + spacing)
        
        # Draw feature bar (darker color to stand out inside group bar)
        ax.bar(feat_bar_x, feat_lr, width=feat_bar_width, color='steelblue', alpha=0.9, 
               edgecolor='white', linewidth=1.5)
        
        # Text directly on top of the bar, centered
        text_x = feat_bar_x + feat_bar_width / 2  # Center of the feature bar
        text_y = feat_lr + max(group_lr_values) * 0.02
        
        ax.text(text_x, text_y,
               f"{label}\nLR={feat_lr:.1f}", 
               ha='center', va='bottom', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Update x-axis to show group names at group bar positions
    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_names, fontsize=11, fontweight='bold')
    
    # Set x-axis limits to prevent text from going outside
    ax.set_xlim(-0.5, len(group_names) - 0.5)
    
    ax.set_ylabel('Likelihood Ratio Statistic', fontsize=12)
    ax.set_title('Feature Significance\n(Main bars = Feature LR, Small bars = Top 2 metrics)', 
                 fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add significance threshold line
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    # Add value labels on main bars - ensure they stay within plot
    max_lr = max(group_lr_values)
    y_margin = max_lr * 0.15  # 15% margin for text
    
    for bar, lr, p in zip(bars, group_lr_values, group_p_values):
        height = bar.get_height()
        label_y = height + max_lr * 0.03
        # Ensure label doesn't go too high
        max_label_y = max_lr * 1.2
        if label_y > max_label_y:
            label_y = max_label_y
        
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
               f'LR={lr:.1f}\np={p:.4f}', ha='center', va='bottom', 
               fontsize=12, fontweight='bold')
    
    # Set y-axis limits
    ax.set_ylim(0, 36)
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='p < 0.001 (***)'),
        Patch(facecolor='#f39c12', label='p < 0.01 (**)'),
        Patch(facecolor='#f1c40f', label='p < 0.05 (*)'),
        Patch(facecolor='#2ecc71', label='p ≥ 0.05 (ns)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('group_significance_barchart.png', dpi=150, bbox_inches='tight')
    print("\nSaved group significance bar chart to 'group_significance_barchart.png'")
    plt.close()

    # ==========================================================================
    # ABLATION STUDY: Drop groups individually and combined
    # ==========================================================================
    print("\n" + "="*60)
    print("ABLATION STUDY: Feature Group Removal")
    print("="*60)
    
    # Baseline: all features
    baseline_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', C=1.0, random_state=42))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    baseline_scores = cross_val_score(baseline_pipeline, X_features, y, cv=cv, scoring='accuracy')
    baseline_acc = baseline_scores.mean()
    
    ablation_results = []
    ablation_results.append({
        'condition': 'All Features (Baseline)',
        'accuracy': baseline_acc,
        'std': baseline_scores.std(),
        'dropped_groups': 'None'
    })
    
    # Test dropping each group individually
    group_drop_tests = {
        'Drop A (Orientation)': 'A_orientation',
        'Drop B (Aperture)': 'B_aperture',
        'Drop C (Setting)': 'C_setting',
        'Drop D (Contact)': 'D_contact',
        'Drop M (Motion)': 'M_motion',
        'Drop A+B+C+D (Keep M only)': ['A_orientation', 'B_aperture', 'C_setting', 'D_contact']
    }
    
    # Test with only top features: A2, B3, C4, D2, M9
    top_feature_labels = ['A2', 'B3', 'C4', 'D2', 'M9']
    label_to_feature = {
        'A2': 'sum_angle',
        'B3': 'aperture_std',
        'C4': 'straightness',
        'D2': 'min_body_distance',
        'M9': 'total_path_length'
    }
    
    top_feature_indices = []
    for label in top_feature_labels:
        feat_name = label_to_feature[label]
        if feat_name in feature_names:
            idx = feature_names.index(feat_name)
            top_feature_indices.append(idx)
    
    # Train on top features only
    if len(top_feature_indices) == 5:
        X_top_features = X_features[:, top_feature_indices]
        pipeline_top = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='linear', C=1.0, random_state=42))
        ])
        scores_top = cross_val_score(pipeline_top, X_top_features, y, cv=cv, scoring='accuracy')
        
        ablation_results.append({
            'condition': 'Only A2+B3+C4+D2+M9 (Top 5 features)',
            'accuracy': scores_top.mean(),
            'std': scores_top.std(),
            'dropped_groups': f'All except {", ".join(top_feature_labels)}',
            'n_features': 5
        })
    
    for test_name, group_key in group_drop_tests.items():
        # Determine which feature indices to keep
        all_indices = set(range(len(feature_names)))
        
        if isinstance(group_key, list):
            # Multiple groups to drop
            indices_to_drop = set()
            for gk in group_key:
                indices_to_drop.update(feature_groups[gk])
        else:
            # Single group to drop
            indices_to_drop = set(feature_groups[group_key])
        
        keep_indices = sorted(list(all_indices - indices_to_drop))
        
        # Train on subset
        X_subset = X_features[:, keep_indices]
        pipeline_subset = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='linear', C=1.0, random_state=42))
        ])
        scores = cross_val_score(pipeline_subset, X_subset, y, cv=cv, scoring='accuracy')
        
        ablation_results.append({
            'condition': test_name,
            'accuracy': scores.mean(),
            'std': scores.std(),
            'dropped_groups': group_key if isinstance(group_key, str) else '+'.join(group_key),
            'n_features': len(keep_indices)
        })
    
    # Create ablation results DataFrame
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df['accuracy_drop'] = baseline_acc - ablation_df['accuracy']
    ablation_df = ablation_df.sort_values('accuracy', ascending=False)
    
    print("\nAblation Results:")
    print(ablation_df.to_string(index=False))
    ablation_df.to_csv('ablation_results.csv', index=False)
    
    # Visualize ablation results
    fig, ax = plt.subplots(figsize=(12, 8))
    
    conditions = ablation_df['condition'].values
    accuracies = ablation_df['accuracy'].values
    stds = ablation_df['std'].values
    accuracy_drops = ablation_df['accuracy_drop'].values
    
    # Color bars: red for large drop, green for small drop
    colors = []
    for drop in accuracy_drops:
        if drop > 0.05:
            colors.append('#e74c3c')  # Red - large drop
        elif drop > 0.02:
            colors.append('#f39c12')  # Orange - medium drop
        elif drop > 0.01:
            colors.append('#f1c40f')  # Yellow - small drop
        else:
            colors.append('#2ecc71')  # Green - minimal drop
    
    bars = ax.barh(range(len(conditions)), accuracies, xerr=stds, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add baseline line
    ax.axvline(x=baseline_acc, color='black', linestyle='--', linewidth=2, 
               label=f'Baseline: {baseline_acc:.3f}')
    
    # Add accuracy drop labels
    for i, (bar, acc, drop) in enumerate(zip(bars, accuracies, accuracy_drops)):
        if drop > 0:
            ax.text(acc + stds[i] + 0.01, i, f'-{drop:.3f}', 
                   va='center', fontsize=9, fontweight='bold', color='red')
    
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions, fontsize=10)
    ax.set_xlabel('Cross-Validation Accuracy', fontsize=12)
    ax.set_title('Ablation Study: Impact of Removing Feature Groups\n(Red=Large drop, Green=Minimal drop)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('ablation_study.png', dpi=150, bbox_inches='tight')
    print("\nSaved ablation study plot to 'ablation_study.png'")
    plt.close()

    print("\n" + "="*60)
    print("PAIRWISE INTERACTION RESULTS (TOP 20)")
    print("="*60)
    interaction_df = results["interaction_results"]
    sig_interactions = interaction_df[interaction_df['significant'] == True]
    print(f"\nSignificant interactions: {len(sig_interactions)}")
    print(interaction_df.head(20).to_string(index=False))

    # Permutation importance results
    if results["permutation_importance"] is not None:
        print("\n" + "="*60)
        print("PERMUTATION IMPORTANCE (from trained SVM)")
        print("="*60)
        perm_df = results["permutation_importance"]
        print(perm_df.to_string(index=False))
        
        # Plot permutation importance
        plt.figure(figsize=(12, 10))
        plt.barh(range(len(perm_df)), perm_df['importance_mean'].values, 
                 xerr=perm_df['importance_std'].values, color='steelblue', alpha=0.8)
        plt.yticks(range(len(perm_df)), perm_df['feature'].values)
        plt.xlabel('Mean Accuracy Decrease')
        plt.title('Permutation Importance for Telicity Prediction')
        plt.tight_layout()
        plt.savefig('permutation_importance.png', dpi=150, bbox_inches='tight')
        print("\nSaved permutation importance plot to 'permutation_importance.png'")
        plt.close()

    # Diagnostic: Why do some high SVM weight features have low univariate significance?
    diagnostic_df = diagnose_svm_vs_stats_discrepancy(
        X_features, y, feature_names, feature_weights, single_df
    )
    diagnostic_df.to_csv('svm_vs_stats_diagnostic.csv', index=False)

    # ==========================================================================
    # AGGREGATED FEATURE IMPORTANCE (sum across significant effects)
    # ==========================================================================
    print("\n" + "="*60)
    print("AGGREGATED FEATURE IMPORTANCE")
    print("(Sum of SVM weights across significant single + interaction effects)")
    print("="*60)
    
    # Initialize aggregated scores
    aggregated_scores = {name: 0.0 for name in feature_names}
    feature_contributions = {name: [] for name in feature_names}  # Track sources
    
    # 1. Add SVM weights for significant single features
    sig_single = single_df[single_df['significant'] == True]
    for _, row in sig_single.iterrows():
        fname = row['feature']
        idx = feature_names.index(fname)
        weight = feature_weights[idx]
        aggregated_scores[fname] += weight
        feature_contributions[fname].append(f"single:{weight:.3f}")
    
    # 2. Add contributions from significant interactions
    sig_interactions = interaction_df[interaction_df['significant'] == True]
    for _, row in sig_interactions.iterrows():
        interaction_name = row['interaction']
        # Parse interaction name (format: "feature1_x_feature2")
        parts = interaction_name.split('_x_')
        if len(parts) == 2:
            f1, f2 = parts[0], parts[1]
            # Get SVM weights for both features
            if f1 in feature_names and f2 in feature_names:
                idx1 = feature_names.index(f1)
                idx2 = feature_names.index(f2)
                w1 = feature_weights[idx1]
                w2 = feature_weights[idx2]
                # Add each feature's weight to its aggregate (when in significant interaction)
                aggregated_scores[f1] += w1
                aggregated_scores[f2] += w2
                feature_contributions[f1].append(f"×{f2}:{w1:.3f}")
                feature_contributions[f2].append(f"×{f1}:{w2:.3f}")
    
    # Sort by aggregated score
    sorted_features = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\nAggregated scores (feature: total from significant effects):")
    print("-" * 50)
    for fname, score in sorted_features:
        if score > 0:
            sources = feature_contributions[fname]
            print(f"  {fname}: {score:.3f}")
            print(f"    Sources: {', '.join(sources[:5])}" + ("..." if len(sources) > 5 else ""))
    
    # Create bar chart with color showing interaction count
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Custom label mapping
    label_mapping = {
        'max_angle': 'A1',
        'sum_angle': 'A2',
        'max_rotation': 'A3',
        'sum_rotation': 'A4',
        'aperture_change': 'B1',
        'aperture_range': 'B2',
        'aperture_std': 'B3',
        'best_region': 'C1',
        'best_dim': 'C2',
        'best_signed_value': 'C3',
        'straightness': 'C4',
        'body_contact_frames': 'D1',
        'min_body_distance': 'D2',
        'body_contact_binary': 'D3',
        'hand_contact_frames': 'D4',
        'min_hand_distance': 'D5',
        'hand_contact_binary': 'D6',
        'mean_velocity': 'M1',
        'velocity_std': 'M2',
        'max_velocity': 'M3',
        'min_velocity': 'M4',
        'mean_acceleration': 'M5',
        'acceleration_std': 'M6',
        'max_acceleration': 'M7',
        'min_acceleration': 'M8',
        'total_path_length': 'M9'
    }
    
    # Filter to features with non-zero scores
    nonzero_features = [(f, s) for f, s in sorted_features if s > 0]
    names = [f[0] for f in nonzero_features]
    display_names = [label_mapping.get(f, f) for f in names]
    scores = [f[1] for f in nonzero_features]
    
    # Count interactions for each feature (sources that start with '×')
    interaction_counts = []
    for fname in names:
        sources = feature_contributions[fname]
        n_interactions = sum(1 for s in sources if s.startswith('×'))
        interaction_counts.append(n_interactions)
    
    # Normalize interaction counts for colormap
    max_interactions = max(interaction_counts) if max(interaction_counts) > 0 else 1
    normalized_counts = [c / max_interactions for c in interaction_counts]
    
    # Color by interaction count (more interactions = warmer color)
    cmap = plt.cm.YlOrRd
    colors = [cmap(0.2 + 0.7 * nc) for nc in normalized_counts]
    
    bars = ax.barh(range(len(names)), scores, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(display_names, fontsize=12, fontweight='bold')
    ax.set_xlabel('Aggregated SVM Weight\n(Sum across significant single + interaction effects)', fontsize=11)
    ax.set_title('Feature Importance: Aggregated Across Significant Effects\n(Color = Number of Significant Interactions)', 
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels with interaction count
    for bar, score, n_int in zip(bars, scores, interaction_counts):
        ax.text(score + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{score:.2f} ({n_int} int.)', va='center', fontsize=9)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_interactions))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Number of Significant Interactions', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('aggregated_feature_importance.png', dpi=150, bbox_inches='tight')
    print("\nSaved aggregated importance plot to 'aggregated_feature_importance.png'")
    plt.close()
    
    # Save to CSV
    agg_df = pd.DataFrame(sorted_features, columns=['feature', 'aggregated_score'])
    agg_df['sources'] = ['; '.join(feature_contributions[f]) for f, _ in sorted_features]
    agg_df.to_csv('aggregated_feature_importance.csv', index=False)

    print("\n=== Analysis Complete ===")
    print("Generated outputs:")
    print("  - new_feature_distributions.png: Feature distribution histograms")
    print("  - new_feature_importance.png: SVM feature importance plot")
    print("  - permutation_importance.png: Permutation importance plot")
    
    # Save results to CSV for further analysis
    single_df.to_csv('single_feature_results.csv', index=False)
    group_df.to_csv('feature_group_results.csv', index=False)
    interaction_df.to_csv('interaction_results.csv', index=False)
    if results["permutation_importance"] is not None:
        results["permutation_importance"].to_csv('permutation_importance.csv', index=False)
    print("\nSaved detailed results to CSV files.")


if __name__ == "__main__":
    main()
