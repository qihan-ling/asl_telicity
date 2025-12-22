import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

from svm import load_holistic_landmarks
from svm import compute_all_features


def analyze_feature_contributions(X_features, y, feature_names, pipeline, filenames):
   """
   Compute feature contributions for each sample.
  
   Feature contribution = scaled_feature_value * svm_weight
   Positive contribution → pushes toward 'telic'
   Negative contribution → pushes toward 'atelic'
   """
   # Extract components
   svm_model = pipeline.named_steps['svm']
   scaler = pipeline.named_steps['scaler']
  
   # Scale features
   X_scaled = scaler.transform(X_features)
  
   # Get SVM weights
   svm_weights = svm_model.coef_[0]
  
   # Compute contributions: scaled_feature * weight
   contributions = X_scaled * svm_weights  # Broadcasting: (n_samples, n_features) * (n_features,)
  
   # Get predictions
   predictions = pipeline.predict(X_features)
   decision_values = pipeline.decision_function(X_features)
  
   # Create detailed analysis
   results = []
  
   for i in range(len(X_features)):
       sample_contrib = contributions[i]
      
       # Top 5 positive contributions (push toward telic)
       top_positive_idx = np.argsort(sample_contrib)[-5:][::-1]
       top_positive = [(feature_names[idx], sample_contrib[idx]) for idx in top_positive_idx]
      
       # Top 5 negative contributions (push toward atelic)
       top_negative_idx = np.argsort(sample_contrib)[:5]
       top_negative = [(feature_names[idx], sample_contrib[idx]) for idx in top_negative_idx]
      
       results.append({
           'sample_idx': filenames[i],
           'true_label': y[i],
           'predicted_label': predictions[i],
           'decision_value': decision_values[i],
           'total_contribution': np.sum(sample_contrib),
           'top_positive': top_positive,
           'top_negative': top_negative,
           'all_contributions': dict(zip(feature_names, sample_contrib))
       })






   # Display results for first 5 samples
   print("\n" + "="*80)
   print("PER-SAMPLE FEATURE CONTRIBUTIONS")
   print("="*80)


   for i in range(len(results)):
       r = results[i]
       print(f"\n{'='*80}")
       print(f"Sample {r['sample_idx']}: True={r['true_label']}, Predicted={r['predicted_label']}")
       print(f"Decision Value: {r['decision_value']:.3f} (positive→telic, negative→atelic)")
       print(f"{'='*80}")
      
       print("\nTop 5 Features Pushing TOWARD Telic:")
       for feat_name, contrib in r['top_positive']:
           print(f"  {feat_name:30s}: +{contrib:.3f}")
      
       print("\nTop 5 Features Pushing TOWARD Atelic:")
       for feat_name, contrib in r['top_negative']:
           print(f"  {feat_name:30s}: {contrib:.3f}")


# Load trained model
with open('/Users/alisonyu/telicity_codes/telicity_model_pipeline.pkl', 'rb') as f:
   pipeline = pickle.load(f)


# Load your training/test data
data_dir = '/Users/alisonyu/telicity_codes/asl_pkl'  # Change this to your data directory




print("Loading data...")
X, y, filenames = load_holistic_landmarks(data_dir)
print(f"Loaded {len(X)} videos")
print(f"Class distribution: {np.unique(y, return_counts=True)}")




print("\nComputing features (with normalization)...")
X_features = np.array([compute_all_features(landmarks) for landmarks in X])


# Feature names (must match training order!)
feature_names = [
   'orientation_change', 'max_rotation', 'aperture_change', 'aperture_range',
   'aperture_std', 'setting_changed', 'num_regions', 'body_contact_binary',
   'body_contact_ratio', 'min_body_distance', 'hand_contact_binary',
   'hand_contact_ratio', 'min_hand_distance', 'path_length', 'displacement',
   'straightness', 'mean_velocity', 'velocity_std', 'max_velocity',
   'min_velocity', 'mean_acceleration', 'acceleration_std', 'max_acceleration',
   'min_acceleration', 'total_path_length', 'num_frames',
]


# Extract weights
svm_model = pipeline.named_steps['svm']
svm_weights = svm_model.coef_[0]


# 1. Print basic weights
print("\n" + "="*80)
print("SVM FEATURE WEIGHTS")
print("="*80)
weight_df = pd.DataFrame({
   'Feature': feature_names,
   'Weight': svm_weights,
   'Abs_Weight': np.abs(svm_weights)
}).sort_values('Abs_Weight', ascending=False)


print(weight_df.to_string(index=False))


# 2. Analyze per-sample contributions
#analyze_feature_contributions(X_features, y, feature_names, pipeline, filenames)

def analyze_each_fold(X_features, y, filenames):
   """
   Perform detailed analysis of each cross-validation fold.
   """
   # Use same CV strategy as training
   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_features, y)):
       print(fold_idx)
       print('Traning index:')
       if len(train_idx) > 1:
           for id in train_idx:
               #print(id)
               print(filenames[id])
       else:
           print(train_idx)
       print('Test index:')
       if len(test_idx) > 1:
           for id in test_idx:
               #print(id)
               print(filenames[id])
       else:
           print(train_idx)
   return 
analyze_each_fold(X_features, y, filenames)
