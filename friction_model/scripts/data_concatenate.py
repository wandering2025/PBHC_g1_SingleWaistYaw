import numpy as np

combined_data_path = '/root/PBHC_g1_SingleWaistYaw/friction_model/data/looser_JumpJumpJump_Randviscous_harder_overturn/iteration_data_combined_24479.npy'
combined_data = np.load(combined_data_path, allow_pickle=True).item()
#dict_keys(['base_angular_vel', 'projected_gravity', 'dof_pos', 'dof_vel', 'dof_angular_acceleration', 'torque', 'friction_coeffs', 'viscous_friction_coeffs'])

base_angular_vel_combined = combined_data['base_angular_vel']
projected_gravity_combined = combined_data['projected_gravity']
dof_pos_combined = combined_data['dof_pos']
dof_vel_combined = combined_data['dof_vel']
dof_angular_acceleration_combined = combined_data['dof_angular_acceleration']
torque_combined = combined_data['torque']
friction_coeffs_combined = combined_data['friction_coeffs']
viscous_friction_coeffs_combined = combined_data['viscous_friction_coeffs']


num_iterations = len(base_angular_vel_combined)
num_dofs = 23


base_angular_vel_cat = np.array(base_angular_vel_combined, dtype=object)
projected_gravity_cat = np.array(projected_gravity_combined, dtype=object)
dof_pos_cat = np.array(dof_pos_combined, dtype=object) 
dof_vel_cat = np.array(dof_vel_combined, dtype=object)
dof_angular_acceleration_cat = np.array(dof_angular_acceleration_combined, dtype=object)
torque_cat = np.array(torque_combined, dtype=object)
friction_coeffs_cat = np.array(friction_coeffs_combined, dtype=object)
viscous_friction_coeffs_cat = np.array(viscous_friction_coeffs_combined, dtype=object)

log_dir = '/root/PBHC_g1_SingleWaistYaw/friction_model/data/looser_JumpJumpJump_Randviscous_harder_overturn/split_cat_data'
base_angular_vel_cat_outpath = f"{log_dir}/{'base_angular_vel_cat.npy'}"
projected_gravity_cat_outpath = f"{log_dir}/{'projected_gravity_cat.npy'}"
dof_pos_cat_outpath = f"{log_dir}/{'dof_pos_cat.npy'}"
dof_vel_cat_outpath = f"{log_dir}/{'dof_vel_cat.npy'}"
dof_angular_acceleration_cat_outpath = f"{log_dir}/{'dof_angular_acceleration_cat.npy'}"
torque_cat_outpath = f"{log_dir}/{'torque_cat.npy'}"
friction_coeffs_cat_outpath = f"{log_dir}/{'friction_coeffs_cat.npy'}"
viscous_friction_coeffs_cat_outpath = f"{log_dir}/{'viscous_friction_coeffs_cat.npy'}"

np.save(base_angular_vel_cat_outpath, base_angular_vel_cat, allow_pickle=True)
np.save(projected_gravity_cat_outpath, projected_gravity_cat, allow_pickle=True)        
np.save(dof_pos_cat_outpath, dof_pos_cat, allow_pickle=True)
np.save(dof_vel_cat_outpath, dof_vel_cat, allow_pickle=True)    
np.save(dof_angular_acceleration_cat_outpath, dof_angular_acceleration_cat, allow_pickle=True)
np.save(torque_cat_outpath, torque_cat, allow_pickle=True)
np.save(friction_coeffs_cat_outpath, friction_coeffs_cat, allow_pickle=True)
np.save(viscous_friction_coeffs_cat_outpath, viscous_friction_coeffs_cat, allow_pickle=True)

print(base_angular_vel_cat.shape)
print(projected_gravity_cat.shape)
print(dof_pos_cat.shape)    
print(dof_vel_cat.shape)
print(dof_angular_acceleration_cat.shape)
print(torque_cat.shape)
print(friction_coeffs_cat.shape)    
print(viscous_friction_coeffs_cat.shape)