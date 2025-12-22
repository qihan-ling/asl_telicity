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

def safe_unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-8)

def compute_palm_normal(hand):
    """Return the unit normal vector of the palm plane."""
    v1 = hand[INDEX_BASE] - hand[WRIST]
    v2 = hand[PINKY_BASE] - hand[WRIST]
    normal = np.cross(v1, v2)
    return safe_unit(normal)

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


def extract_orientation_features(palm_normals, forearm_dirs, ref_palm_normal):
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

    return {
        "max_angle": max_angle,
        "sum_angle": sum_angle,
        "max_rotation": max_rotation,
        "sum_rotation": sum_rotation
    }

# --- 2. Setting CHANGE ---

#Setting change within a major body region
import numpy as np

# Regions you want to track
FOCUS_REGIONS = ["above_head", "head", "torso", "below_torso"]
DIMENSIONS = ["side", "vertical", "depth"]

nose_y = np.mean(pose_frames[:, NOSE, 1])
shoulder_y = np.mean(
    pose_frames[:, LEFT_SHOULDER, 1] + pose_frames[:, RIGHT_SHOULDER, 1]) / 2)
hip_y = np.mean((pose_frames[:, LEFT_HIP, 1] +
                   pose_frames[:, RIGHT_HIP, 1]) / 2)

def compute_signed_change(positions):
    """
    Compute signed movement change along each axis for one body region.
    Movements in opposite directions cancel out.
    positions: (T×3) XYZ coordinates over time.
    Returns a dict mapping dimension → signed sum.
    """
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

        for dim in DIMENSIONS:
            signed_val = signed_vals[dim]
            strength = abs(signed_val)

            if strength > best_strength:
                best_strength = strength
                best_signed_value = signed_val
                best_region = region_name
                best_dim = dim

    # Create final 8-dim feature vector
    vec = np.zeros(8, dtype=float)

    if best_region is not None:
        vec[FOCUS_REGIONS.index(best_region)] = 1.0

    if best_dim is not None:
        vec[4 + DIMENSIONS.index(best_dim)] = 1.0

    # final slot = signed change value (positive or negative)
    vec[7] = best_signed_value

    return vec

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