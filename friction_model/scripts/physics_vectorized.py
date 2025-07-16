import numpy as np
import torch

combined_data_path = '/root/PBHC_g1_SingleWaistYaw/friction_model/data/LOOSE_JumpJumpJump_RandViscous_Hard_iteration_data_combined_20315.npy'
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

joint_radius = np.array([0.025, 0.025, 0.025, 0.025, 0.008, 0.006,
            0.025, 0.025, 0.025, 0.025, 0.008, 0.006,
            0.025,
            0.025, 0.025, 0.025, 0.025, 0.018,
            0.025, 0.025, 0.025, 0.025, 0.018])

link_inertia_path = '/root/PBHC_g1_SingleWaistYaw/friction_model/data/rigid_body_inertia.npy'
link_inertia_data = np.load(link_inertia_path, allow_pickle=True).item()
link_inertia = [0]*23  # Initialize an array for joint inertia
#dict_keys(['pelvis', 
# 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
# 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', 
# 'torso_link', 
# 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_roll_rubber_hand',
# 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_rubber_hand'])
link_inertia[0] = link_inertia_data['left_hip_pitch_link']
link_inertia[1] = link_inertia_data['left_hip_roll_link']
link_inertia[2] = link_inertia_data['left_hip_yaw_link']
link_inertia[3] = link_inertia_data['left_knee_link']
link_inertia[4] = link_inertia_data['left_ankle_pitch_link']
link_inertia[5] = link_inertia_data['left_ankle_roll_link']
link_inertia[6] = link_inertia_data['right_hip_pitch_link']
link_inertia[7] = link_inertia_data['right_hip_roll_link']
link_inertia[8] = link_inertia_data['right_hip_yaw_link']
link_inertia[9] = link_inertia_data['right_knee_link'] 
link_inertia[10] = link_inertia_data['right_ankle_pitch_link']
link_inertia[11] = link_inertia_data['right_ankle_roll_link']
link_inertia[12] = link_inertia_data['pelvis']
link_inertia[13] = link_inertia_data['left_shoulder_pitch_link']
link_inertia[14] = link_inertia_data['left_shoulder_roll_link']
link_inertia[15] = link_inertia_data['left_shoulder_yaw_link']
link_inertia[16] = link_inertia_data['left_elbow_link']
link_inertia[17] = link_inertia_data['left_wrist_roll_rubber_hand']
link_inertia[18] = link_inertia_data['right_shoulder_pitch_link']
link_inertia[19] = link_inertia_data['right_shoulder_roll_link']   
link_inertia[20] = link_inertia_data['right_shoulder_yaw_link']
link_inertia[21] = link_inertia_data['right_elbow_link']
link_inertia[22] = link_inertia_data['right_wrist_roll_rubber_hand']

link_inertia = np.array(link_inertia)

def physics_dof_vel_vectorized(f_c, r, F_trans, J, dof_ang_acc, torque_a, tao_load, f_v, dof_vel):
    """
    使用 NumPy 矩阵运算高效地预测关节速度。

    参数均为 NumPy 数组:
    - f_c, dof_ang_acc, torque_a, f_v, dof_vel: 形状为 (steps, joints) 的二维数组
    - r, F_trans, J, tao_load: 形状为 (joints,) 的一维数组，将通过广播机制参与运算
    """
    sign_dof_vel = np.sign(dof_vel)
    numerator = -f_c * r * F_trans * sign_dof_vel - J * dof_ang_acc + torque_a + tao_load
    dof_vel_pre = numerator / f_v
    return dof_vel_pre


def differentiable_physics_model_torch(f_c, r, F_trans, J, dof_ang_acc, torque_a, tao_load, f_v, dof_vel):
    """
    Differentiable physics model implemented in PyTorch.
    All inputs are expected to be PyTorch Tensors.
    """
    sign_dof_vel = torch.sign(dof_vel)
    
    # Broadcasting will apply 1D parameters (r, F_trans, etc.) to the 2D tensors
    numerator = -f_c * r * F_trans * sign_dof_vel - J * dof_ang_acc + torque_a + tao_load
    
    # Add a small epsilon to the denominator to prevent division by zero
    dof_vel_pre = numerator / (f_v + 1e-8)
    
    return dof_vel_pre





def calculate_physics(i, F_trans_out, J_out, tao_load_out):
    # i indicates steps in one iteration
    #for i in range(len(dof_vel_combined)):

    dof_pos_mat = dof_pos_combined[i]
    dof_vel_mat = dof_vel_combined[i]
    dof_angular_mat = dof_angular_acceleration_combined[i]
    torque_mat = torque_combined[i]
    coulomb_friction_mat = friction_coeffs_combined[i]
    viscous_friction_mat = viscous_friction_coeffs_combined[i]

    dof_vel_pre_mat = physics_dof_vel_vectorized(
        f_c = coulomb_friction_mat,
        r = joint_radius, 
        F_trans = F_trans_out,
        J = J_out,
        dof_ang_acc = dof_angular_mat,
        torque_a = torque_mat,
        tao_load = tao_load_out,
        f_v = viscous_friction_mat,
        dof_vel = dof_vel_mat
    )

    print(f'Iteration {i}: Physics calculated dof_vel {dof_vel_pre_mat.shape}:\n {dof_vel_pre_mat}')

def main():

    num_dofs = 23

    #for j in range(len(dof_vel_combined)):
    for j in range(2):
        num_steps = len(dof_vel_combined[j])

        F_trans_out = np.random.randn(num_steps, num_dofs)
        J_out = np.random.randn(num_steps, num_dofs)
        tao_load_out = np.random.randn(num_steps, num_dofs)
        calculate_physics(j, F_trans_out, J_out, tao_load_out)

if __name__ == "__main__":
    main()