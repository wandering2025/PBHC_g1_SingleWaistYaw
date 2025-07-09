nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_AllFalse \
+rewards=motion_tracking/main \
experiment_name=dev_iteration_record \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/131_07_poses_RootLifted_0.06_Ankle_Roll_Clipped_Wrist00.pkl" \
seed=1 \
+device=cuda:0 > nohup_dev_iteration_record_isaacgym 2>&1 &


*/root/PBHC_g1_SingleWaistYaw/humanoidverse/envs/legged_base_task/legged_robot_base.py*

# get inertia and angular acceleration in-real-time

result:

Inertia of left_hip_pitch_link 
tensor([[ 3.3308e-03,  3.7447e-05, -3.9121e-05],

        [ 3.7447e-05,  2.3799e-03,  1.7510e-04],

        [-3.9121e-05,  1.7510e-04,  2.4646e-03]], device='cuda:0')
(kg·m²)

angular_acceleration of left_hip_pitch_link:
tensor([[ 3.5783e+02,  1.7112e+03, -1.5390e+01],
 (rad/s²)



*/root/PBHC_g1_SingleWaistYaw/humanoidverse/envs/legged_base_task/legged_robot_base.py*

 <!-- obs_buf = np.concatenate((self.action, base_ang_vel.flatten(), dof_pos, dof_vel, history_obs_buf, projected_gravity, [self.ref_motion_phase]), axis=-1, dtype=np.float32) -->

# get obs elements in-real-time

result: [num_envs=4]

angular_acceleration of left_hip_pitch_link:
tensor([[ -215.5218,  1797.7745, -1018.6453],
        [  668.3995,   304.6680,  -295.3296],
        [   44.5415,  1619.7920,   733.3596],
        [  138.1279,   369.3927,  -226.4175]], device='cuda:0')
base_angular_vel: tensor([[ 2.6256,  7.0565, -4.9048],
        [ 3.2470, -2.0003, -1.2365],
        [-0.2506, -0.9352,  3.9745],
        [ 0.9766,  0.3821, -0.9837]], device='cuda:0')
projected gravity: tensor([[ 0.2087, -0.0285, -0.9776],
        [ 0.0717,  0.0056, -0.9974],
        [ 0.2001,  0.0346, -0.9792],
        [ 0.1848, -0.0084, -0.9827]], device='cuda:0')
dof_pos:tensor([[-6.2001e-01, -5.0935e-03,  4.6928e-02,  5.2712e-01, -1.0198e-01,
         -3.1240e-03, -1.3106e+00,  3.2034e-01, -1.2978e-01,  7.3326e-01,
          5.4336e-02, -9.0590e-03, -1.2642e-01, -6.2508e-04,  4.9711e-02,
         -4.1039e-01,  1.2977e-01, -8.2747e-02,  3.4125e-01, -6.4381e-02,
          1.5745e-02,  2.1864e-01,  2.4702e-01],
        [ 6.0693e-02,  4.5503e-01, -8.1417e-02,  2.0903e-01, -2.0643e-01,
          2.7891e-02, -4.6914e-01, -1.7599e-01,  2.4820e-01,  6.6595e-01,
          8.8346e-03, -1.4347e-02, -3.5898e-02,  4.1064e-02,  1.0710e-02,
         -2.2500e-01,  4.2710e-01, -7.3074e-02, -2.3925e-01, -1.3468e+00,
         -1.6404e-01,  2.9891e-01,  3.2823e-01],
        [-8.4876e-01,  2.1139e-01, -1.7199e-01,  5.2035e-01, -9.2997e-02,
          4.2671e-03, -2.4550e-01, -3.0804e-01,  1.9047e-01,  7.2081e-01,
         -1.1867e-01, -1.9773e-02,  2.6024e-02,  2.5121e-02,  1.8332e-01,
         -1.2958e-01,  1.1361e-01, -6.1793e-02,  2.1964e-01, -4.3896e-02,
         -1.1521e-02,  2.6097e-01,  3.0039e-01],
        [-3.0016e-01, -9.1662e-02,  2.9748e-02,  2.4280e-01, -1.4336e-01,
          3.0279e-02, -3.1293e-01,  7.6481e-02,  1.6401e-02,  2.7310e-01,
         -1.2723e-01, -1.2636e-02,  6.0393e-03,  7.0818e-02,  2.6289e-02,
         -1.3122e-01,  4.1714e-01, -2.8815e-02,  3.3673e-01, -2.4391e-02,
         -2.1760e-02,  4.8519e-01,  3.2435e-01]], device='cuda:0')
dof_vel:tensor([[ 1.7156e+00,  7.9544e-01,  3.1788e+00, -1.5518e+01, -5.2461e+00,
          2.9525e+00, -3.9484e+00, -2.1078e+00, -6.1186e+00, -1.5548e+01,
         -1.1767e+00,  1.1291e+00,  1.0833e+01, -2.2176e+00, -1.1804e+01,
         -9.1701e-01, -4.7166e+00,  4.0228e-01,  1.5590e+01,  3.3869e+00,
         -8.9871e+00, -1.1266e+00,  1.3634e+01],
        [ 3.8971e+00, -6.2124e+00,  2.9237e+00,  2.0115e+00,  2.4986e+00,
          2.5955e+00,  7.4088e+00, -5.3267e+00, -2.6606e+00, -1.5869e+01,
         -2.0309e+00, -4.6769e-01,  2.8745e+00, -3.8324e+00, -1.7814e+00,
          4.9047e-01, -1.0920e+01,  4.2760e+00,  1.9559e+01,  6.4653e+00,
          4.2620e+00, -9.3325e+00,  1.2647e+01],
        [ 8.8868e+00, -6.1341e-01,  6.8666e+00, -1.6291e+01, -3.3765e+00,
         -8.6842e-01,  1.2874e+01, -3.4687e+00, -4.5454e+00, -1.3593e+01,
         -3.9825e+00,  1.6376e+00, -3.5613e+00,  9.7819e-01, -5.8240e+00,
          2.2602e+00,  8.0781e-01, -1.3182e+00,  1.6725e+01,  2.0959e+00,
         -3.2813e+00, -6.9264e-01,  1.6745e+01],
        [ 1.4203e+00, -2.1747e-01,  2.0597e+00, -1.2100e-01, -3.5900e+00,
          4.9241e-01,  1.0965e+00, -1.3672e+00, -6.4369e-01, -4.8399e-01,
         -3.9923e+00, -4.0392e-03,  2.4460e+00, -1.9702e+00, -2.8707e+00,
         -8.6363e-02, -7.9998e+00,  6.4744e+00,  1.2428e+01,  4.1731e-01,
         -1.4446e+00, -9.0134e+00,  1.5714e+01]], device='cuda:0')



*/root/PBHC_g1_SingleWaistYaw/humanoidverse/envs/legged_base_task/legged_robot_base.py*
*/root/PBHC_g1_SingleWaistYaw/humanoidverse/agents/mh_ppo/mh_ppo.py*
# added record code for: 
#   rigidbody-inertias, 
#   link_angular_acceleration, 
#   base_angular_vel
#   projected_gravity
#   dof_pos
#   dof_vel

rigid_body_inertia.npy

iteration_{}_env0_data.npy
(steps, 24, 3)



nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_AllFalse \
+rewards=motion_tracking/main \
experiment_name=ForwardLeanWalk__iteration_record \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/74_02_poses_RootLifted_0.08_Ankle_Roll_Clipped_Wrist00.pkl" \
seed=1 \
+device=cuda:0 > nohup_ForwardLeanWalk_iteration_record 2>&1 &


*/root/PBHC_g1_SingleWaistYaw/humanoidverse/simulator/isaacgym/isaacgym.py*
# SET f_viscous = 0.05 for each joint




**record log**
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_AllFalse \
+rewards=motion_tracking/main \
experiment_name=moon_walk_iteration_record \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/moon_walk_RootLifted_0.06.pkl" \
seed=1 \
+device=cuda:0 > nohup_moon_walk_iteration_record 2>&1 &
