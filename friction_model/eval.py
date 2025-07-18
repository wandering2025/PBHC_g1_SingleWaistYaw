import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import joblib
import sys

# ==============================================================================
# 1. MODEL DEFINITIONS (Copy from your training script)
#    - We need these classes to rebuild the model structure before loading weights.
# ==============================================================================

class ResidualBlock(nn.Module):
    """
    A residual block with a skip connection.
    If input and output dimensions are different, a linear layer
    is used on the shortcut path to match the dimensions.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float = 0.1):
        super(ResidualBlock, self).__init__()
        self.main_path = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main_path(x) + self.shortcut(x))

class ParameterNetResNet(nn.Module):
    """
    The same dynamic ResNet-based model definition from your training script.
    """
    def __init__(self, input_dim: int, block_dims: list, num_dofs: int = 23, dropout_rate: float = 0.1):
        super(ParameterNetResNet, self).__init__()
        self.num_dofs = num_dofs
        first_block_dim = block_dims[0]
        self.initial_layer = nn.Sequential(
            nn.Linear(input_dim, first_block_dim),
            nn.BatchNorm1d(first_block_dim),
            nn.ReLU()
        )
        res_layers = []
        current_dim = first_block_dim
        for h_dim in block_dims:
            res_layers.append(ResidualBlock(current_dim, h_dim, dropout_rate))
            current_dim = h_dim
        self.residual_blocks = nn.Sequential(*res_layers)
        self.output_head = nn.Linear(current_dim, num_dofs * 3)

    def forward(self, x: torch.Tensor):
        x = self.initial_layer(x)
        x = self.residual_blocks(x)
        raw_params = self.output_head(x)
        f_trans_raw, j_raw, tao_load_raw = torch.split(raw_params, self.num_dofs, dim=1)
        F_trans = f_trans_raw
        tao_load = tao_load_raw
        J = F.softplus(torch.tanh(j_raw))
        return F_trans, J, tao_load

def differentiable_physics_model(f_c, r, F_trans, J, dof_ang_acc, torque_a, tao_load, sign_dof_vel):
    """
    The modified physics model that returns the numerator for stable loss calculation.
    """
    numerator = -f_c * r * F_trans * sign_dof_vel - J * dof_ang_acc + torque_a + tao_load
    return numerator

# ==============================================================================
# 2. EVALUATION SCRIPT CONFIGURATION
# ==============================================================================

# --- IMPORTANT: CONFIGURE YOUR PATHS AND PARAMETERS HERE ---

# Path to the specific experiment directory you want to evaluate.
# Example: '/root/PBHC_g1_SingleWaistYaw/friction_model/logs/20250716_181040_dev_resnet...'
# You can also pass this as a command line argument.
EXPERIMENT_DIR = '/root/PBHC_g1_SingleWaistYaw/friction_model/logs/20250718_120930_dev_resnet_Predict_dof_vel_lr1e-3'

# Path to the validation data
DATA_PATH = '/root/PBHC_g1_SingleWaistYaw/friction_model/data/LOOSE_JumpJumpJump_RandViscous_Hard_concat_data/data_for_training/val_data.npy'

# Model parameters (MUST MATCH THE TRAINED MODEL)
MODEL_PARAMS = {
    'input_dim': 98,
    'num_dofs': 23,
    'block_dims': [512, 1024,1024, 256], # Example for a 3-block network, change if needed
    'dropout_rate': 0.1
}

# Input keys used for constructing the network input
INPUT_KEYS = sorted(['base_angular_vel', 'projected_gravity', 'dof_pos', 'dof_vel', 'dof_angular_acceleration', 'torque'])

# Number of specific examples to print out
NUM_EXAMPLES_TO_PRINT = 5

# ==============================================================================
# 3. MAIN EVALUATION LOGIC
# ==============================================================================

def main():
    """
    Main function to run the evaluation.
    """
    # Allow overriding experiment dir from command line
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = EXPERIMENT_DIR
        
    print(f"--- Starting Evaluation for Experiment ---")
    print(f"Log Directory: {log_dir}\n")

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    model_path = os.path.join(log_dir, 'best_params_predictor.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
        
    model = ParameterNetResNet(**MODEL_PARAMS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")

    # --- Load Scaler ---
    scaler_path = os.path.join(log_dir, 'x_scaler.gz')
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file not found at {scaler_path}")
        return
    x_scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully.")

    # --- Load Data ---
    val_data = np.load(DATA_PATH, allow_pickle=True).item()
    num_trajectories = len(val_data[INPUT_KEYS[0]])
    print(f"Validation data loaded. Found {num_trajectories} trajectories.\n")
    
    # Use a fixed tensor for 'r' on the correct device
    r_tensor = torch.tensor([
        0.025, 0.025, 0.025, 0.025, 0.008, 0.006, 0.025, 0.025, 0.025, 0.025, 0.008, 0.006, 0.025, 
        0.025, 0.025, 0.025, 0.025, 0.018, 0.025, 0.025, 0.025, 0.025, 0.018
    ], dtype=torch.float32).to(device)

    # For cleaner printing of numpy arrays
    np.set_printoptions(precision=4, suppress=True)

    # --- 1. Print Specific Examples ---
    print(f"--- Displaying {NUM_EXAMPLES_TO_PRINT} Specific Examples ---")
    # We will take samples from the first trajectory in the validation set
    trajectory_idx = 0 
    
    # Get all data for the chosen trajectory
    trajectory_data = {key: torch.from_numpy(val_data[key][trajectory_idx].astype(np.float32)).to(device) for key in val_data.keys()}
    
    # Prepare the raw, unscaled input for the NN
    x_unscaled_traj = np.concatenate([val_data[key][trajectory_idx] for key in INPUT_KEYS], axis=1)
    
    for i in range(min(NUM_EXAMPLES_TO_PRINT, len(x_unscaled_traj))):
        print(f"\n--- [ Sample {i+1} from Trajectory {trajectory_idx+1} ] ---")
        
        # Prepare a single timestep input
        x_unscaled_sample = x_unscaled_traj[i:i+1] # Shape (1, 98)
        x_scaled_sample = torch.from_numpy(x_scaler.transform(x_unscaled_sample)).float().to(device)
        
        with torch.no_grad():
            # Perform inference
            F_trans_pred, J_pred, tao_load_pred = model(x_scaled_sample)

            # Get ground truth values for this timestep
            dof_vel_gt = trajectory_data['dof_vel'][i:i+1]
            sign_dof_vel_gt = torch.sign(dof_vel_gt)
            dof_acc_gt = trajectory_data['dof_angular_acceleration'][i:i+1]
            torque_gt = trajectory_data['torque'][i:i+1]
            f_c_gt = trajectory_data['friction_coeffs'][i:i+1]
            f_v_gt = trajectory_data['viscous_friction_coeffs'][i:i+1]

            # Calculate predicted numerator using the physics model
            numerator_predicted = differentiable_physics_model(
                f_c=f_c_gt, r=r_tensor, F_trans=F_trans_pred, J=J_pred,
                dof_ang_acc=dof_acc_gt, torque_a=torque_gt,
                tao_load=tao_load_pred, sign_dof_vel=sign_dof_vel_gt
            )
            
            # Calculate the target numerator from ground truth
            target_numerator = f_v_gt * dof_vel_gt

        # Print results (slicing to show first 4 values for readability)
        s = 4 # slice size
        print(f"  Inputs (GT):")
        print(f"    - Dof Vel      (rad/s) : {dof_vel_gt.cpu().numpy()[0, :s]}")
        print(f"    - Dof Acc (rad/s^2) : {dof_acc_gt.cpu().numpy()[0, :s]}")
        print(f"    - Torque (Nm)         : {torque_gt.cpu().numpy()[0, :s]}")

        print(f"  Predicted Parameters:")
        print(f"    - Inertia J            : {J_pred.cpu().numpy()[0, :s]}")
        print(f"    - Force F_trans        : {F_trans_pred.cpu().numpy()[0, :s]}")
        print(f"    - Load tao_load        : {tao_load_pred.cpu().numpy()[0, :s]}")
        print(f"  Comparison:")

        epsilon = 1e-8
        predicted_dof_vel = numerator_predicted.cpu().numpy()[0, :s] / (f_v_gt.cpu().numpy()[0, :s] + epsilon)
        print(f"    - f_v  : {f_v_gt.cpu().numpy()[0, :s]}")

        print(f"    - Predicted Numerator  : {numerator_predicted.cpu().numpy()[0, :s]}")
        print(f"    - Target Numerator     : {target_numerator.cpu().numpy()[0, :s]}")


        print(f"    - Predicted Dof Vel  : {predicted_dof_vel}")
        print(f"    - Target Dof Vel     : {dof_vel_gt.cpu().numpy()[0, :s]}")
        #print(f"    - Predicted Numerator  : {numerator_predicted.cpu().numpy()[0, :s]}")
        #print(f"    - Target Numerator     : {target_numerator.cpu().numpy()[0, :s]}")

    # --- 2. Calculate Overall Loss on the Entire Validation Set ---
    print("\n\n--- Calculating Overall Loss on Validation Set ---")
    total_val_loss = 0
    total_samples = 0
    criterion = nn.MSELoss(reduction='sum') # Use sum to aggregate loss, then average manually

    with torch.no_grad():
        for traj_idx in range(num_trajectories):
            # Prepare data for one full trajectory
            x_unscaled = np.concatenate([val_data[key][traj_idx] for key in INPUT_KEYS], axis=1)
            x_scaled = torch.from_numpy(x_scaler.transform(x_unscaled)).float().to(device)
            
            # Get all ground truth tensors for the trajectory
            traj_data_tensors = {key: torch.from_numpy(val_data[key][traj_idx].astype(np.float32)).to(device) for key in val_data.keys()}
            sign_vel = torch.sign(traj_data_tensors['dof_vel'])

            # Inference
            F_trans, J, tao_load = model(x_scaled)
            
            # Prediction
            pred_numerator = differentiable_physics_model(
                f_c=traj_data_tensors['friction_coeffs'], r=r_tensor, F_trans=F_trans, J=J,
                dof_ang_acc=traj_data_tensors['dof_angular_acceleration'], torque_a=traj_data_tensors['torque'],
                tao_load=tao_load, sign_dof_vel=sign_vel
            )

            # Target
            target_dof_vel = traj_data_tensors['dof_vel']
            #target_num = traj_data_tensors['viscous_friction_coeffs'] * traj_data_tensors['dof_vel']
            pred_dof_vel = pred_numerator / (traj_data_tensors['viscous_friction_coeffs'] + 1e-8)

            loss = criterion(pred_dof_vel, target_dof_vel)
            total_val_loss += loss.item()
            total_samples += len(x_unscaled)
            
    avg_val_loss = total_val_loss / total_samples
    print(f"\nOverall Validation MSE Loss for DOF_VEL: {avg_val_loss:.6f}")
    print("--- Evaluation Finished ---")

if __name__ == '__main__':
    main()