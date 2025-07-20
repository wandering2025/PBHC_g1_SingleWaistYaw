import numpy as np
import os

# --- 1. Setup Paths and Constants ---
# Define the source directory for the split data and the destination for the processed training data.
log_dir = '/root/PBHC_g1_SingleWaistYaw/friction_model/data/looser_JumpJumpJump_Randviscous_harder_overturn/split_cat_data'
save_dir = '/root/PBHC_g1_SingleWaistYaw/friction_model/data/looser_JumpJumpJump_Randviscous_harder_overturn/data_for_training'
os.makedirs(save_dir, exist_ok=True)

# Define the fixed length of the sequence to be sampled from each iteration.
# For any iteration with more than 20 steps, a continuous sequence of 20 will be randomly sampled.
FIXED_SEQ_LENGTH = 20

# --- 2. Load All Necessary Data ---
# All data keys that are needed for physics modeling and training.
data_keys = [
    'base_angular_vel', 'projected_gravity', 'dof_pos', 'dof_vel',
    'dof_angular_acceleration', 'torque', 'friction_coeffs', 'viscous_friction_coeffs'
]
all_data = {}
print("Loading data from .npy files...")
for key in data_keys:
    file_path = os.path.join(log_dir, f"{key}_cat.npy")
    # Load data as object arrays since they are ragged (arrays of arrays with different lengths).
    all_data[key] = np.load(file_path, allow_pickle=True)

# --- 3. Process Data: Truncate/Sample to Fixed Length ---
# This crucial step ensures that every iteration has a uniform sequence length,
# making it suitable for batching in deep learning models.
print(f"Processing data: sampling a continuous sequence of length {FIXED_SEQ_LENGTH} from each iteration...")
num_iterations = all_data['dof_pos'].shape[0]
np.random.seed(42) # Set seed for reproducibility of random sampling.

# First, determine a single random start index for each iteration.
# This ensures that for a given iteration, all data types (dof_pos, dof_vel, etc.) are sliced identically,
# maintaining their temporal alignment.
start_indices = np.zeros(num_iterations, dtype=int)
for i in range(num_iterations):
    # Get the original sequence length (number of steps) for the current iteration using 'dof_pos' as a reference.
    original_seq_length = all_data['dof_pos'][i].shape[0]
    
    # If the sequence is longer than our target length, choose a random starting point for the slice.
    if original_seq_length > FIXED_SEQ_LENGTH:
        # The latest possible start index is (original_length - fixed_length).
        max_start_index = original_seq_length - FIXED_SEQ_LENGTH
        start_indices[i] = np.random.randint(0, max_start_index + 1)
    # If the sequence is already the desired length or shorter, we start from index 0.
    # Given the problem description (min length is 20), this will apply to length=20 cases.
    else:
        start_indices[i] = 0

# Now, create new processed data arrays which are dense NumPy arrays, not object arrays.
processed_data = {}
for key in data_keys:
    # For each iteration 'i', create a slice using the pre-determined start index.
    # This creates a list of sliced numpy arrays, where each array has the shape (FIXED_SEQ_LENGTH, num_features).
    sliced_list = [
        all_data[key][i][start_indices[i] : start_indices[i] + FIXED_SEQ_LENGTH]
        for i in range(num_iterations)
    ]
    # Stack the list of arrays into a single, dense numpy array along a new axis (axis=0).
    # The new shape will be (num_iterations, FIXED_SEQ_LENGTH, num_features).
    processed_data[key] = np.stack(sliced_list, axis=0)
    print(f"  - Processed '{key}' with final shape: {processed_data[key].shape}")

# --- 4. Shuffle and Split Indices at the Iteration Level ---
# This ensures that data from the same original robot run stays together 
# in the same split (train/val/test), preventing data leakage.
print("\nShuffling and splitting iteration indices...")
# Use the same random seed for the train/val/test split for reproducibility.
np.random.seed(42)
indices = np.arange(num_iterations)
np.random.shuffle(indices)

# Split indices into 80% train, 10% validation, 10% test.
train_idx, val_idx, test_idx = np.split(indices, [int(num_iterations*0.8), int(num_iterations*0.9)])

# --- 5. Create and Save Split Datasets ---
# Create dictionaries for each split containing the processed, dense data.
print("Creating and saving train, validation, and test set dictionaries...")

train_data = {key: processed_data[key][train_idx] for key in data_keys}
val_data = {key: processed_data[key][val_idx] for key in data_keys}
test_data = {key: processed_data[key][test_idx] for key in data_keys}

# Save the final dictionaries. Each .npy file now contains a dictionary of dense arrays.
np.save(os.path.join(save_dir, 'train_data.npy'), train_data)
np.save(os.path.join(save_dir, 'val_data.npy'), val_data)
np.save(os.path.join(save_dir, 'test_data.npy'), test_data)

# --- 6. Verification ---
print("\nData preparation for physics identification complete! ")
print(f"Saved training data with {len(train_idx)} iterations.")
print(f"Saved validation data with {len(val_idx)} iterations.")
print(f"Saved test data with {len(test_idx)} iterations.")
print(f"Each iteration now has a fixed sequence length of {FIXED_SEQ_LENGTH}.")
print(f"Data saved in: {save_dir}")

# Optional: Verify the shape of one of the arrays in the saved training data to confirm the process worked.
verify_train_data = np.load(os.path.join(save_dir, 'train_data.npy'), allow_pickle=True).item()
print(f"\nVerification check: Shape of 'dof_pos' in train_data: {verify_train_data['dof_pos'].shape}")