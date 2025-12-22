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
      Np  = project_onto_plane(N,  F)
      N0p = project_onto_plane(N0, F)

      # normalize projections
      Np  /= np.linalg.norm(Np)
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
         rotation_values.append(abs(rotation))   # "either direction" → use magnitude
         

      # compute the four required features
      max_angle     = np.max(angle_values)
      sum_angle     = np.sum(angle_values)

      max_rotation  = np.max(rotation_values)
      sum_rotation  = np.sum(rotation_values)

      return max_angle, sum_angle, max_rotation, sum_rotation

   
   max_angle, sum_angle, max_rotation, sum_rotation = extract_orientation_features(palm_normals, forearm_dirs)

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
   shoulder_y = np.mean(pose_frames[:, LEFT_SHOULDER, 1] + pose_frames[:, RIGHT_SHOULDER, 1] / 2)
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
      #vec = np.zeros(8, dtype=float)

      if best_region is not None:
         #vec[FOCUS_REGIONS.index(best_region)] = 1.0
         best_region = FOCUS_REGIONS.index(best_region)

      if best_dim is not None:
         #vec[4 + DIMENSIONS.index(best_dim)] = 1.0
         best_dim = DIMENSIONS.index(best_dim)

      # final slot = signed change value (positive or negative)
      #vec[7] = best_signed_value

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
      
   regions = {'above_head':[], 'head':[], 'torso':[], 'below_torso':[]}
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

import itertools
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from sklearn.inspection import permutation_importance


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
    X = np.asarray(X)
    y = np.asarray(y)

    # null model: intercept only
    X_null = sm.add_constant(np.ones(len(y)))
    model_null = sm.Logit(y, X_null).fit(disp=False)

    # full model
    X_full = sm.add_constant(X)
    model_full = sm.Logit(y, X_full).fit(disp=False)

    lr_stat = 2 * (model_full.llf - model_null.llf)
    df = 26
    p_value = sm.stats.chisqprob(lr_stat, df)

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

def run_full_statistical_pipeline(X, y, feature_groups, feature_names=None, model=None):
    """
    feature_groups: dict mapping group name → list of feature indices
    model: optional ML model for permutation importance
    """

    # 1) Standardize
    X_scaled, scaler = standardize_features(X)

    # 2) Single features
    single_feature_results = test_single_features(X_scaled, y, feature_names)

    # 3) Feature groups
    group_results = []
    for group_name, indices in feature_groups.items():
        df = test_feature_group(X_scaled, y, indices, feature_names)
        df["group_name"] = group_name
        group_results.append(df)
    group_results = pd.concat(group_results).sort_values("p")

    # 4) Interactions
    interaction_results = test_all_interactions(X_scaled, y, feature_names)


    return {
        "scaled_features": X_scaled,
        "single_feature_results": single_feature_results,
        "group_results": group_results,
        "interaction_results": interaction_results,
        "scaler": scaler
    }


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
       # Phonological features (0-14)
       'max_angle',     
       'sum_angle',
       'max_rotation',          
       'sum_rotation',
       'aperture_change',       
       'aperture_range',        
       'aperture_std',           
       'best_region', 
       'best_dim', 
       'best_signed_value',
       'body_contact_frames',    
       'min_body_distance',      
       'body_contact_binary',      
       'hand_contact_frames',    
       'min_hand_distance',
       'hand_contact_binary',      
       'straightness',
       # Motion features (15-23)
       'mean_velocity',
       'velocity_std',
       'max_velocity',
       'min_velocity',
       'mean_acceleration',
       'acceleration_std',
       'max_acceleration',
       'min_acceleration',
       'total_path_length'
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
   print("\n=== Statistical Analysis of ALL Features ===")
   from scipy.stats import mannwhitneyu


   # for i, name in enumerate(feature_names):  # Only phonological features
   #     telic_values = X_features[y == 'telic', i]
   #     atelic_values = X_features[y == 'atelic', i]


   #     # Mann-Whitney U test (non-parametric)
   #     statistic, p_value = mannwhitneyu(
   #         telic_values, atelic_values, alternative='two-sided')


   #     print(f"\n{name}:")
   #     print(
   #         f"  Telic:   mean={np.mean(telic_values):.3f}, std={np.std(telic_values):.3f}")
   #     print(
   #         f"  Atelic:  mean={np.mean(atelic_values):.3f}, std={np.std(atelic_values):.3f}")
   #     print(
   #         f"  p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")

   # ================================================================
   # AUTOMATED ANALYSIS DRIVER
   # ================================================================

   import pandas as pd
   import itertools


   # ================================================================
   # 1. AUTO-GENERATE FEATURE GROUPS
   #    - groups of size 1 handled separately in single-feature tests
   #    - here we make groups of size 2 up to max group size
   # ================================================================

   feature_indices = list(range(26))

   feature_groups = {}

   # Create groups of size 2 → 5 (change as needed)
   MAX_GROUP_SIZE = min(5,26)

   group_id = 0
   for group_size in range(2, MAX_GROUP_SIZE + 1):
      for combo in itertools.combinations(feature_indices, group_size):
         label = f"group_{group_id}_" + "_".join(str(i) for i in combo)
         feature_groups[label] = list(combo)
         group_id += 1


   # ================================================================
   # 2. RUN FULL PIPELINE
   # ================================================================
   results = run_full_statistical_pipeline(
      X=X,
      y=y,
      feature_groups=feature_groups,
      feature_names=feature_names,
      model=None   # supply trained SVM model if you want permutation importance
   )


   # ================================================================
   # 3. PRINT SUMMARIES
   # ================================================================
   print("\n===== SINGLE FEATURE RESULTS =====")
   print(results["single_feature_results"])

   print("\n===== FEATURE GROUP RESULTS (TOP 20) =====")
   print(results["group_results"].head(20))

   print("\n===== INTERACTION RESULTS (TOP 20) =====")
   print(results["interaction_results"].head(20))

   print("\n=== Analysis Complete ===")
   print("Generated outputs:")
   print("  - feature_distributions.png: Feature distribution histograms")
   print("  - feature_importance.png: SVM feature importance plot")




if __name__ == "__main__":
   main()
