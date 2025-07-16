import numpy as np
import os

# --- 1. Setup Paths ---
log_dir = '/root/PBHC_g1_SingleWaistYaw/friction_model/data/LOOSE_JumpJumpJump_RandViscous_Hard_concat_data/split_cat_data'
save_dir = '/root/PBHC_g1_SingleWaistYaw/friction_model/data/LOOSE_JumpJumpJump_RandViscous_Hard_concat_data/data_for_training'
os.makedirs(save_dir, exist_ok=True)

# --- 2. Load All Necessary Data ---
# All data keys that are needed for the physics formula and loss calculation
data_keys = [
    'base_angular_vel', 'projected_gravity', 'dof_pos', 'dof_vel',
    'dof_angular_acceleration', 'torque', 'friction_coeffs', 'viscous_friction_coeffs'
]
all_data = {}
print("Loading data from .npy files...")
for key in data_keys:
    file_path = os.path.join(log_dir, f"{key}_cat.npy")
    all_data[key] = np.load(file_path, allow_pickle=True)

# --- 3. Shuffle and Split Indices ---
num_iterations = all_data['dof_pos'].shape[0]
np.random.seed(42)
indices = np.arange(num_iterations)
np.random.shuffle(indices)

train_idx, val_idx, test_idx = np.split(indices, [int(num_iterations*0.8), int(num_iterations*0.9)])

# --- 4. Create and Save Split Datasets ---
# We no longer create X and y, but save a single dictionary for each split
print("Creating and saving train, validation, and test set dictionaries...")

train_data = {key: all_data[key][train_idx] for key in data_keys}
val_data = {key: all_data[key][val_idx] for key in data_keys}
test_data = {key: all_data[key][test_idx] for key in data_keys}

np.save(os.path.join(save_dir, 'train_data.npy'), train_data)
np.save(os.path.join(save_dir, 'val_data.npy'), val_data)
np.save(os.path.join(save_dir, 'test_data.npy'), test_data)

# --- 5. Verification ---
print("\nData preparation for physics identification complete!")
print(f"Saved training data with {len(train_idx)} iterations.")
print(f"Saved validation data with {len(val_idx)} iterations.")
print(f"Saved test data with {len(test_idx)} iterations.")
print(f"Data saved in: {save_dir}")