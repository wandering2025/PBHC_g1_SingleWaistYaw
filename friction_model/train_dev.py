import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import os
import datetime
import sys
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1. Parameter Prediction Network (Flexible and Well-Commented) ---
class ParameterNet(nn.Module):
    """
    A flexible MLP that predicts physical parameters from state vectors.
    """
    def __init__(self, input_dim: int, hidden_dims: list, num_dofs: int = 23, dropout_rate: float = 0.1):
        """
        Initializes the network.
        :param input_dim: Dimension of the input feature vector (e.g., 98).
        :param hidden_dims: A list defining the number of neurons in each hidden layer. 
                            The length of the list determines the depth of the network.
                            Example: [1024, 512, 256] creates a 3-hidden-layer network.
        :param num_dofs: Number of degrees of freedom (e.g., 23).
        :param dropout_rate: Dropout probability.
        """
        super(ParameterNet, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # Dynamically build the hidden layers based on the hidden_dims list
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim # Update the dimension for the next layer
            
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output head predicts the raw values for the 3 parameters (F_trans, J, tao_load)
        # Total output dimension is 3 * num_dofs
        self.output_head = nn.Linear(current_dim, num_dofs * 3)
        self.num_dofs = num_dofs

    def forward(self, x: torch.Tensor):
        # Input x shape: [B, input_dim], where B is the total number of time steps in the batch.
        
        # Pass through the hidden layers
        x = self.hidden_layers(x)
        
        # Predict the raw parameter values
        # raw_params shape: [B, num_dofs * 3]
        raw_params = self.output_head(x)
        
        # Split the raw output into three groups for each parameter
        # Each group will have a shape of [B, num_dofs]
        f_trans_raw, j_raw, tao_load_raw = torch.split(raw_params, self.num_dofs, dim=1)
        
        # --- Final Parameter Generation ---
        # F_trans and tao_load can be positive or negative, so we use the direct linear output.
        F_trans = f_trans_raw
        tao_load = tao_load_raw
        
        # J (Inertia) must be physically positive.
        # We use softplus to ensure the output is always > 0, which stabilizes training.
        # No scaling is applied, allowing the model to learn the magnitude freely.
        #J = torch.softplus(j_raw) 

        #J = F.softplus(j_raw)
        J = F.softplus(torch.tanh(j_raw))

        
        # Return the final parameter tensors for the physics model
        # Each has a shape of [B, num_dofs]
        return F_trans, J, tao_load

# --- 2. Differentiable Physics Model ---
def differentiable_physics_model(f_c, r, F_trans, J, dof_ang_acc, torque_a, tao_load, f_v, dof_vel):
    """
    Differentiable physics model implemented in PyTorch.
    All inputs are expected to be PyTorch Tensors.
    """
    sign_dof_vel = torch.sign(dof_vel)
    numerator = -f_c * r * F_trans * sign_dof_vel - J * dof_ang_acc + torque_a + tao_load
    dof_vel_pre = numerator / (f_v + 1e-8)
    return dof_vel_pre

# --- 3. Custom Dataset with Standardization ---
class PhysicsDataset(Dataset):
    def __init__(self, data: dict, input_keys: list, x_scaler: StandardScaler):
        self.data = data
        self.input_keys = sorted(input_keys) # Ensure consistent key order
        self.all_keys = sorted(list(data.keys()))
        self.num_iterations = len(data[self.all_keys[0]])
        self.x_scaler = x_scaler

    def __len__(self):
        return self.num_iterations

    def __getitem__(self, idx: int):
        # Concatenate raw input features for a single iteration
        x_iter_raw = np.concatenate([self.data[key][idx] for key in self.input_keys], axis=1, dtype=np.float32)
        
        # Apply standardization to the network inputs
        x_iter_scaled = self.x_scaler.transform(x_iter_raw)
        
        # Prepare a dictionary of all data needed for the physics-based loss
        physics_data = {key: torch.from_numpy(self.data[key][idx].astype(np.float32)) for key in self.all_keys}
        # Add the scaled network input to the dictionary
        physics_data['nn_input'] = torch.from_numpy(x_iter_scaled)
        
        return physics_data

# --- 4. Custom Collate Function ---
def collate_fn_physics(batch):
    # This function takes a list of dictionary samples and collates them into a single batch dictionary
    collated_batch = {}
    keys = batch[0].keys()
    for key in keys:
        tensors = [item[key] for item in batch]
        collated_batch[key] = torch.cat(tensors, dim=0)
    return collated_batch

# --- 5. Training Loop ---
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, session_save_dir, num_epochs, patience):
    best_model_path = os.path.join(session_save_dir, 'best_params_predictor.pth')
    min_val_loss = float('inf')
    epochs_no_improve = 0

    # Create the fixed 'r' tensor once and move it to the device
    r_tensor = torch.tensor([
        0.025, 0.025, 0.025, 0.025, 0.008, 0.006, 
        0.025, 0.025, 0.025, 0.025, 0.008, 0.006, 
        0.025, 
        0.025, 0.025, 0.025, 0.025, 0.018, 
        0.025, 0.025, 0.025, 0.025, 0.018
    ], dtype=torch.float32).to(device)

    print(f"\n--- Starting Training: State-Dependent Parameter Prediction ---")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch_on_device = {k: v.to(device) for k, v in batch.items()}
            nn_input = batch_on_device['nn_input']
            
            optimizer.zero_grad()
            
            # Step 1: Predict parameters from the current state
            F_trans, J, tao_load = model(nn_input)
            
            # Step 2: Use predicted parameters in the physics model
            dof_vel_predicted = differentiable_physics_model(
                f_c=batch_on_device['friction_coeffs'], r=r_tensor, F_trans=F_trans, J=J,
                dof_ang_acc=batch_on_device['dof_angular_acceleration'], torque_a=batch_on_device['torque'],
                tao_load=tao_load, f_v=batch_on_device['viscous_friction_coeffs'], dof_vel=batch_on_device['dof_vel']
            )
            
            # Step 3: Compute loss against ground truth velocity
            loss = criterion(dof_vel_predicted, batch_on_device['dof_vel'])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_on_device = {k: v.to(device) for k, v in batch.items()}
                nn_input = batch_on_device['nn_input']
                F_trans, J, tao_load = model(nn_input)
                dof_vel_predicted = differentiable_physics_model(
                    f_c=batch_on_device['friction_coeffs'], r=r_tensor, F_trans=F_trans, J=J,
                    dof_ang_acc=batch_on_device['dof_angular_acceleration'], torque_a=batch_on_device['torque'],
                    tao_load=tao_load, f_v=batch_on_device['viscous_friction_coeffs'], dof_vel=batch_on_device['dof_vel']
                )
                loss = criterion(dof_vel_predicted, batch_on_device['dof_vel'])
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.1e}")

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best validation loss. Model saved.")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break
        scheduler.step(avg_val_loss)

    print("\n--- Training Finished ---")
    print(f"Best model saved at: {best_model_path}")

# --- 6. Main Execution Block ---
def main():
    # --- Configuration ---
    input_keys = ['base_angular_vel', 'projected_gravity', 'dof_pos', 'dof_vel', 'dof_angular_acceleration', 'torque']
    input_dim = sum([3, 3, 23, 23, 23, 23]) # 98
    num_dofs = 23
    
    # ***** Network Architecture Configuration *****
    # Easily change network depth and width by modifying this list.
    # e.g., [1024, 512, 256] for a 3-hidden-layer network.
    # e.g., [2048, 2048, 1024, 512] for a 4-hidden-layer network.
    hidden_dims = [1024, 2048, 1024, 512] 
    
    # Training Hyperparameters
    learning_rate = 1e-4
    weight_decay = 1e-6
    batch_size = 1024
    num_epochs = 1000
    patience = 150
    scheduler_patience = 40

    # --- Paths and Session Management ---
    log_base_dir = '/root/PBHC_g1_SingleWaistYaw/friction_model/logs'
    data_dir = '/root/PBHC_g1_SingleWaistYaw/friction_model/data/LOOSE_JumpJumpJump_RandViscous_Hard_concat_data/data_for_training'
    
    experiment_name = 'dev_friction_lr1e-5_batch1024' # Default experiment name
    for arg in sys.argv:
        if arg.startswith('+experiment='):
            experiment_name = arg.split('=')[1]
            break
            
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_save_dir = os.path.join(log_base_dir, f"{timestamp}_{experiment_name}")
    os.makedirs(session_save_dir, exist_ok=True)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Standardization ---
    print("Loading pre-split data...")
    train_data = np.load(os.path.join(data_dir, 'train_data.npy'), allow_pickle=True).item()
    val_data = np.load(os.path.join(data_dir, 'val_data.npy'), allow_pickle=True).item()

    print("Fitting StandardScaler on training data...")
    sorted_input_keys = sorted(input_keys)
    # Concatenate all time steps from all training iterations to fit the scaler
    full_train_x_list = [np.concatenate([train_data[key][i] for key in sorted_input_keys], axis=1) for i in range(len(train_data[input_keys[0]]))]
    full_train_x = np.concatenate(full_train_x_list, axis=0)
    
    x_scaler = StandardScaler()
    x_scaler.fit(full_train_x)
    del full_train_x, full_train_x_list # Free up memory
    
    # Save the fitted scaler for inference
    joblib.dump(x_scaler, os.path.join(session_save_dir, 'x_scaler.gz'))
    print(f"Input feature scaler (x_scaler) saved to {session_save_dir}")

    # --- Create Datasets and DataLoaders ---
    train_dataset = PhysicsDataset(train_data, input_keys, x_scaler)
    val_dataset = PhysicsDataset(val_data, input_keys, x_scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_physics)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_physics)

    # --- Initialize Model, Criterion, and Optimizer ---
    model = ParameterNet(input_dim=input_dim, 
                         hidden_dims=hidden_dims, 
                         num_dofs=num_dofs, 
                         dropout_rate=0.0).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=scheduler_patience, verbose=True)
    
    print("\n--- Starting run for experiment:", experiment_name, "---")
    print(f"Network Architecture (hidden_dims): {hidden_dims}")
    print("\nModel Structure:")
    print(model)
    
    # --- Start Training ---
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, session_save_dir, num_epochs, patience)

if __name__ == '__main__':
    main()