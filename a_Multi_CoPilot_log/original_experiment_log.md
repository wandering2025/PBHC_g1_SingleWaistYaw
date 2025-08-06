nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main \
experiment_name=_ori_reward \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/" \
seed=1 \
+device=cuda:0 > nohup__ori_reward 2>&1 &


################################## 7.26 / 23:05 #################

[32.26] 
## killed at 50000 ##
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main \
experiment_name=shoulders_poses_ori_reward \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/shoulders_poses_root_lifted_0.03.pkl" \
seed=1 \
+device=cuda:0 > nohup_shoulders_poses_ori_reward 2>&1 &


[32.26] 
## killed at 50000 ##
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main \
experiment_name=stretches_poses_ori_reward \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/stretches_poses_root_lifted_0.03.pkl" \
seed=1 \
+device=cuda:0 > nohup_stretches_poses_ori_reward 2>&1 &

[111] 7.27 / 09:41
 
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main \
experiment_name=misc_7_poses_ori_reward \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/misc_7_poses_RootLifted_0.018.pkl" \
seed=1 \
+device=cuda:0 > nohup_misc_7_poses_ori_reward 2>&1 &

[111]  
## --killed at 50000-- ##
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main \
experiment_name=jumpjumpjump_ori_reward \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/01_01_poses_root_lifted_0.03.pkl" \
seed=1 \
+device=cuda:0 > nohup_jumpjumpjump_ori_reward 2>&1 &


[32322]
## --killed at 44000-- ##
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main \
experiment_name=UpRightWalk_ori_reward \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/02_01_poses_root_lifted_0.055_inter0.5_S0-30_E86-30.pkl" \
seed=1 \
+device=cuda:0 > nohup_UpRightWalk_ori_reward 2>&1 &


[32322]
## --killed at 43000-- ##
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main \
experiment_name=GentleWalk_ori_reward \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/123_01_poses_root_lifted_0.04.pkl" \
seed=1 \
+device=cuda:0 > nohup_GentleWalk_ori_reward 2>&1 &



[54044]
## --killed at 47000 --#
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main \
experiment_name=ForwardLeanWalk_ori_reward \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/74_02_poses_RootLifted_0.025_inter0.5_S0-30_E133-30.pkl" \
seed=1 \
+device=cuda:0 > nohup_ForwardLeanWalk_ori_reward 2>&1 &

-------23:26--------
[54044] 
## --killed at 47000 --#
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main \
experiment_name=TurnJump_ori_reward \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/75_07_poses_RootLifted_0.05.pkl" \
seed=1 \
+device=cuda:0 > nohup_TurnJump_ori_reward 2>&1 &






################################## 7.28 / 15:12 #################
[32.26] ## --killed at 45000 --#
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main \
experiment_name=catching_and_throwing_ori_reward \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/0015_catching_and_throwing_stageii_RootLifted_0.05.pkl" \
seed=1 \
+device=cuda:0 > nohup_catching_and_throwing_ori_reward 2>&1 &


[111]
# killed at 55000
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main \
experiment_name=side_circle_walk_ori_reward \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/54_22_poses_RootLifted_0.03.pkl" \
seed=1 \
+device=cuda:0 > nohup_side_circle_walk_ori_reward 2>&1 &


[32322] 
# killed at 49000
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main \
experiment_name=CasualDance_ori_reward \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/13_14_poses.pkl" \
seed=1 \
+device=cuda:0 > nohup_CasualDance_ori_reward 2>&1 &


[54044]

nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main \
experiment_name=SquatDown_ori_reward \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/26_10_poses_RootLifted_0.06.pkl" \
seed=1 \
+device=cuda:0 > nohup_SquatDown_ori_reward 2>&1 &





##########-----------7.29 10:30-----------############

QkWalk1_stageii_RootLifted_0.025_inter0.5_S0-30_E77-30.pkl
[111]
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main \
experiment_name=QkWalk_inter_ori_reward \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/QkWalk1_stageii_RootLifted_0.025_inter0.5_S0-30_E77-30.pkl" \
seed=1 \
+device=cuda:0 > nohup_QkWalk_inter_ori_reward 2>&1 &

##############-----------7.29 14:13------#########################
0026_kicking1_stageii_RootLifted_0.04.pkl
[38902]
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main \
experiment_name=FrontKick_ori_reward \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/0026_kicking1_stageii_RootLifted_0.04.pkl" \
seed=1 \
+device=cuda:0 > nohup_FrontKick_ori_reward 2>&1 &



#########-----------7.29 / 23:35-----------#################

EricCamper04_stageii_RootLifted_0.03.pkl
[54044]
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main \
experiment_name=EricCamper_Taiji_ori_reward \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/EricCamper04_stageii_RootLifted_0.03.pkl" \
seed=1 \
+device=cuda:0 > nohup_EricCamper_Taiji_ori_reward 2>&1 &




############----------------7.30 11:27-----------###########
90_05_poses_RootLifted_0.015.pkl #SpinKick
[111]
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main \
experiment_name=SpinKick_ori_reward \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/90_05_poses_RootLifted_0.015.pkl" \
seed=1 \
+device=cuda:0 > nohup_SpinKick_ori_reward 2>&1 &





