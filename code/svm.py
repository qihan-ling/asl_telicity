#This script is generated with the asistance of AI
import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
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


   # --- 1. ORIENTATION CHANGE ---
   # Use wrist → middle finger direction
   def get_hand_orientation(hand):
       vector = hand[MIDDLE_TIP] - hand[WRIST]
       return vector / (np.linalg.norm(vector) + 1e-8)


   start_orientation = get_hand_orientation(hand_frames[0])
   end_orientation = get_hand_orientation(hand_frames[-1])


   cos_angle = np.clip(np.dot(start_orientation, end_orientation), -1.0, 1.0)
   orientation_change = np.degrees(np.arccos(cos_angle))


   # Maximum rotation between consecutive frames
   max_rotation = 0
   for i in range(1, len(hand_frames)):
       curr_orient = get_hand_orientation(hand_frames[i])
       prev_orient = get_hand_orientation(hand_frames[i-1])
       cos_angle = np.clip(np.dot(curr_orient, prev_orient), -1.0, 1.0)
       rotation = np.degrees(np.arccos(cos_angle))
       max_rotation = max(max_rotation, rotation)


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


   # --- 3. SETTING CHANGE (Body Region) ---
   # Define body regions based on y-coordinates
   nose_y = np.mean(pose_frames[:, NOSE, 1])
   shoulder_y = np.mean(
       (pose_frames[:, LEFT_SHOULDER, 1] + pose_frames[:, RIGHT_SHOULDER, 1]) / 2)
   hip_y = np.mean((pose_frames[:, LEFT_HIP, 1] +
                   pose_frames[:, RIGHT_HIP, 1]) / 2)


   def classify_region(wrist_y):
       # Thresholds adjusted for normalized coordinates
       if wrist_y < nose_y - 0.1:  # above head
           return "above_head"
       elif wrist_y < shoulder_y:  # head region
           return "head"
       elif wrist_y < hip_y:  # torso
           return "torso"
       else:  # lower region
           return "lower"


   regions_visited = set()
   for frame in pose_frames:
       wrist_y = frame[wrist_idx, 1]
       region = classify_region(wrist_y)
       regions_visited.add(region)


   setting_changed = 1 if len(regions_visited) > 1 else 0
   num_regions = len(regions_visited)


   # --- 4. CONTACT DETECTION ---
   # A. Hand-to-body contact
   body_landmarks = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
   # Threshold adjusted for normalized coordinates (relative to torso size)
   body_contact_threshold = 0.25


   body_contact_detected = False
   body_contact_frames = 0
   min_body_distance = float('inf')


   for frame in pose_frames:
       wrist_pos = frame[wrist_idx, :3]  # x, y, z
       for body_idx in body_landmarks:
           body_pos = frame[body_idx, :3]
           distance = np.linalg.norm(wrist_pos - body_pos)
           min_body_distance = min(min_body_distance, distance)


           if distance < body_contact_threshold:
               body_contact_detected = True
               body_contact_frames += 1


   body_contact_binary = 1 if body_contact_detected else 0
   body_contact_ratio = body_contact_frames / len(pose_frames)


   # B. Hand-to-hand contact
   hand_contact_threshold = 0.20  # adjusted for normalized coordinates
   hand_contact_detected = False
   hand_contact_frames = 0
   min_hand_distance = float('inf')


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
                hand_contact_detected = True
                hand_contact_frames += 1
    
        hand_contact_ratio = hand_contact_frames / len(both_hands_frames)
   else:
        hand_contact_ratio = 0.0
        min_hand_distance = 10.0

#    if len(both_hands_frames) > 0:
#        for left_hand, right_hand, _ in both_hands_frames:
#            # Check distance between any landmarks of left and right hands
#            for left_lm in left_hand:
#                for right_lm in right_hand:
#                    distance = np.linalg.norm(left_lm - right_lm)
#                    min_hand_distance = min(min_hand_distance, distance)


#                    if distance < hand_contact_threshold:
#                        hand_contact_detected = True
#                        hand_contact_frames += 1
#                        break  # Found contact in this frame
#                if distance < hand_contact_threshold:
#                    break


#        hand_contact_ratio = hand_contact_frames / len(both_hands_frames)
#    else:
#        # If only one hand visible throughout, no hand-to-hand contact possible
#        hand_contact_ratio = 0.0
#        min_hand_distance = 10.0


   hand_contact_binary = 1 if hand_contact_detected else 0
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
       orientation_change,      # 0: hand rotation (start to end)
       max_rotation,           # 1: maximum rotation between frames
       aperture_change,        # 2: fingertip spread change
       aperture_range,         # 3: range of aperture
       aperture_std,           # 4: aperture variability
       setting_changed,        # 5: binary (crossed body regions)
       num_regions,            # 6: count of regions visited
       body_contact_binary,    # 7: binary hand-body contact detected
       body_contact_ratio,     # 8: proportion of frames with hand-body contact
       min_body_distance,      # 9: closest approach to body
       hand_contact_binary,    # 10: binary hand-hand contact detected
       hand_contact_ratio,     # 11: proportion of frames with hand-hand contact
       min_hand_distance,      # 12: closest distance between hands
       path_length,            # 13: total wrist movement
       displacement,           # 14: start-to-end wrist distance
       straightness,           # 15: movement efficiency
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
       return np.zeros(10)  # fallback


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
       len(pose_frames),             # 9: number of frames
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
   plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
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
   plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
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
           pose_frames = [f['pose'] for f in landmarks if f['pose'] is not None]
           right_hand_frames = [f['right_hand'] for f in landmarks if f['right_hand'] is not None]
           left_hand_frames = [f['left_hand'] for f in landmarks if f['left_hand'] is not None]
          
           print(f"  Total frames: {len(landmarks)}")
           print(f"  Pose frames: {len(pose_frames)}")
           #print(f"  Right hand frames: {right_hand_frames}")
           #print(f"  Left hand frames: {left_hand_frames}")
          
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
               torso_height = np.linalg.norm((left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2)
               torso_size = (shoulder_width + torso_height) / 2
              
               print(f"  Torso size: {torso_size:.6f}")
               print(f"  Shoulder width: {shoulder_width:.6f}")
               print(f"  Torso height: {torso_height:.6f}")



def main():
   """
   Main pipeline for telicity detection.
   """
   # Configuration
   data_dir = '/Users/alisonyu/telicity_codes/asl_pkl'  # Change this to your data directory


   print("Loading data...")
   X, y, filenames = load_holistic_landmarks(data_dir)
   print(f"Loaded {len(X)} videos")
   print(f"Class distribution: {np.unique(y, return_counts=True)}")
   diagnose_invalid_features(X, y, filenames)


   print("\nComputing features (with normalization)...")
   X_features = np.array([compute_all_features(landmarks) for landmarks in X])
   print(f"Feature matrix shape: {X_features.shape}")


   # Feature names
   feature_names = [
       # Phonological features (0-15)
       'orientation_change',
       'max_rotation',
       'aperture_change',
       'aperture_range',
       'aperture_std',
       'setting_changed',
       'num_regions',
       'body_contact_binary',
       'body_contact_ratio',
       'min_body_distance',
       'hand_contact_binary',        # NEW
       'hand_contact_ratio',         # NEW
       'min_hand_distance',          # NEW
       'path_length',
       'displacement',
       'straightness',
       # Motion features (16-25)
       'mean_velocity',
       'velocity_std',
       'max_velocity',
       'min_velocity',
       'mean_acceleration',
       'acceleration_std',
       'max_acceleration',
       'min_acceleration',
       'total_path_length',
       'num_frames',
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
   #print("\nVisualizing feature distributions...")
   #visualize_features(X_features, y, feature_names)


   # Train and evaluate SVM
   print("\nTraining SVM classifier...")
   feature_weights, pipeline = train_and_evaluate_svm(X_features, y)
   #feature_weights, pipeline = train_and_evaluate_svm(X, y)


   # Analyze feature importance
   analyze_feature_importance(feature_weights, feature_names)


   # Statistical analysis of phonological features
   print("\n=== Statistical Analysis of Phonological Features ===")
   from scipy.stats import mannwhitneyu


   for i, name in enumerate(feature_names[:16]):  # Only phonological features
       telic_values = X_features[y == 'telic', i]
       atelic_values = X_features[y == 'atelic', i]


       # Mann-Whitney U test (non-parametric)
       statistic, p_value = mannwhitneyu(
           telic_values, atelic_values, alternative='two-sided')


       print(f"\n{name}:")
       print(
           f"  Telic:   mean={np.mean(telic_values):.3f}, std={np.std(telic_values):.3f}")
       print(
           f"  Atelic:  mean={np.mean(atelic_values):.3f}, std={np.std(atelic_values):.3f}")
       print(
           f"  p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")


   print("\n=== Analysis Complete ===")
   print("Generated outputs:")
   print("  - feature_distributions.png: Feature distribution histograms")
   print("  - feature_importance.png: SVM feature importance plot")




if __name__ == "__main__":
   main()
