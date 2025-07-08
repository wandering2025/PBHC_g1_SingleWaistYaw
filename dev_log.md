nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=main \
+rewards=motion_tracking/main \
experiment_name=dev_inertia_AngularAcceleration \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/131_07_poses_RootLifted_0.06_Ankle_Roll_Clipped_Wrist00.pkl" \
seed=1 \
+device=cuda:0 > nohup_dev_inertia_AngularAcceleration 2>&1 &


/root/PBHC_g1_SingleWaistYaw/humanoidverse/envs/legged_base_task/legged_robot_base.py

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


