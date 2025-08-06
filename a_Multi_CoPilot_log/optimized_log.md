nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/ \
experiment_name=_opt_reward_grok \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/" \
seed=1 \
+device=cuda:0 > nohup__opt_reward_grok 2>&1 &


[38902]  ## --killed at 42000 --#
# main_opti_Gemini2.5pro_jumpjumpjump 
# with 30000+ training log
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_opti_Gemini2.5pro_jumpjumpjump \
experiment_name=jumpjumpjump_reward_opt \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/01_01_poses_root_lifted_0.03.pkl" \
seed=1 \
+device=cuda:0 > nohup_jumpjumpjump_reward_opt 2>&1 &


[38902] ---killed at 60000--
# main_opti_grok3_20000_jumpjumpjump
# with 20000 training log
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_opti_grok3_20000_jumpjumpjump \
experiment_name=jumpjumpjump_reward_opt_grok \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/01_01_poses_root_lifted_0.03.pkl" \
seed=1 \
+device=cuda:0 > nohup_jumpjumpjump_reward_opt_grok 2>&1 &


[111] 7.28 / 21:00 --killed at 46000--
# with 20000 training log + video description

#main_opt_grok3_video_misc_20000
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_opt_grok3_video_misc_20000 \
experiment_name=misc_7_poses_opt_reward_grok \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/misc_7_poses_RootLifted_0.018.pkl" \
seed=1 \
+device=cuda:0 > nohup_misc_7_poses_opt_reward_grok 2>&1 &


[54044] # 7.28/ 23:50  
main_opt_grok3_video_TurnJump.yaml
# with 20000 training log + video description
# *killed bad* #
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_opt_grok3_video_TurnJump \
experiment_name=TurnJump_opt_reward_grok \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/75_07_poses_RootLifted_0.05.pkl" \
seed=1 \
+device=cuda:0 > nohup_TurnJump_opt_reward_grok 2>&1 &


[32322]

main_opt_grok_20000_UpRightWalk
# with 20000 training log 
# killed at 40000
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_opt_grok_20000_UpRightWalk \
experiment_name=UpRightWalk_opt_reward_grok \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/02_01_poses_root_lifted_0.055_inter0.5_S0-30_E86-30.pkl" \
seed=1 \
+device=cuda:0 > nohup_UpRightWalk_opt_reward_grok 2>&1 &


######----------7.29 13:46------------##########

main_grok_opt_stretches_poses_20000
[32.26] 
# with 20000 training log 
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_grok_opt_stretches_poses_20000 \
experiment_name=stretches_poses_opt_reward_grok \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/stretches_poses_root_lifted_0.03.pkl" \
seed=1 \
+device=cuda:0 > nohup_stretches_poses_opt_reward_grok 2>&1 &



main_grok_opt_catching_and_throwing_20000
[32.26]
# with 20000 training log 
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_grok_opt_catching_and_throwing_20000 \
experiment_name=catching_and_throwing_opt_reward_grok \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/0015_catching_and_throwing_stageii_RootLifted_0.05.pkl" \
seed=1 \
+device=cuda:0 > nohup_catching_and_throwing_opt_reward_grok 2>&1 &


[54044] 7.26 / 16:51
# with 20000 training log 
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_opt_grok3_NOvideo_TurnJump \
experiment_name=TurnJump_opt_reward_grok_Novideo \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/75_07_poses_RootLifted_0.05.pkl" \
seed=1 \
+device=cuda:0 > nohup_TurnJump_opt_reward_grok_Novideo 2>&1 &




[32322] 7.30 / 11:00
123_01_poses_root_lifted_0.04.pkl
main_grok_40000_ExtraGestureDescrip_GentleWalk
# grok fed with training log 40000 and description of unnatual gesture
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_grok_40000_ExtraGestureDescrip_GentleWalk \
experiment_name=GentleWalk_opt_reward_grok_ExtraUnnatual \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/123_01_poses_root_lifted_0.04.pkl" \
seed=1 \
+device=cuda:0 > nohup_GentleWalk_opt_reward_grok_ExtraUnnatual 2>&1 &


7.30 / 13:31
54_22_poses_RootLifted_0.03.pkl #side_circle_walk
[38902]
# with 20000 training log 
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_grok_opt_side_circle_walk \
experiment_name=side_circle_walk_opt_reward_grok \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/54_22_poses_RootLifted_0.03.pkl" \
seed=1 \
+device=cuda:0 > nohup_side_circle_walk_opt_reward_grok 2>&1 &


[32.26]
# with 17000 training log
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_grok_opt_17000_QkWalk \
experiment_name=QkWalk1_inter_opt_reward_grok_17000 \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/QkWalk1_stageii_RootLifted_0.025_inter0.5_S0-30_E77-30.pkl" \
seed=1 \
+device=cuda:0 > nohup_QkWalk1_inter_opt_reward_grok_17000 2>&1 &




[54044]
FrontKick
main_grok20000_opt_front_kick_ExtraMotionDesc

nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_grok20000_opt_front_kick_ExtraMotionDesc \
experiment_name=FrontKick_opt_reward_grok20000_ExtraMotionDesc \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/0026_kicking1_stageii_RootLifted_0.04.pkl" \
seed=1 \
+device=cuda:0 > nohup_FrontKick_opt_reward_grok20000_ExtraMotionDesc 2>&1 &


7.31 / 14:51
[32322]
main_grok20000_opt_SpinKick
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_grok20000_opt_SpinKick \
experiment_name=SpinKick_opt_reward_grok \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/90_05_poses_RootLifted_0.015.pkl" \
seed=1 \
+device=cuda:0 > nohup_SpinKick_opt_reward_grok 2>&1 &

[32.26]
main_grok_opt_misc_7
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_grok_opt_misc_7 \
experiment_name=misc_7_opt_reward_grok \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/misc_7_poses_RootLifted_0.018.pkl" \
seed=1 \
+device=cuda:0 > nohup_misc_7_opt_reward_grok 2>&1 &



# ################################### itration experiment ##################
[32.26] 7.31 22:17

#main_grok_opt_CatchThrow_4000

nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_grok_opt_CatchThrow_4000 \
experiment_name=catching_and_throwing_opt_reward_grok_4000 \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/0015_catching_and_throwing_stageii_RootLifted_0.05.pkl" \
seed=1 \
+device=cuda:0 > nohup_catching_and_throwing_opt_reward_grok_4000 2>&1 &


[111]
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_grok_opt_CatchThrow_8000 \
experiment_name=catching_and_throwing_opt_reward_grok_8000 \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/0015_catching_and_throwing_stageii_RootLifted_0.05.pkl" \
seed=1 \
+device=cuda:0 > nohup_catching_and_throwing_opt_reward_grok_8000 2>&1 &


[111]
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_grok_opt_CatchThrow_12000 \
experiment_name=catching_and_throwing_opt_reward_grok_12000 \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/0015_catching_and_throwing_stageii_RootLifted_0.05.pkl" \
seed=1 \
+device=cuda:0 > nohup_catching_and_throwing_opt_reward_grok_12000 2>&1 &


[38902]
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_grok_opt_CatchThrow_16000 \
experiment_name=catching_and_throwing_opt_reward_grok_16000 \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/0015_catching_and_throwing_stageii_RootLifted_0.05.pkl" \
seed=1 \
+device=cuda:0 > nohup_catching_and_throwing_opt_reward_grok_16000 2>&1 &


[32322]
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_grok_opt_CatchThrow_24000 \
experiment_name=catching_and_throwing_opt_reward_grok_24000 \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/0015_catching_and_throwing_stageii_RootLifted_0.05.pkl" \
seed=1 \
+device=cuda:0 > nohup_catching_and_throwing_opt_reward_grok_24000 2>&1 &


[54044]
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/main_grok_opt_CatchThrow_28000 \
experiment_name=catching_and_throwing_opt_reward_grok_28000 \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/0015_catching_and_throwing_stageii_RootLifted_0.05.pkl" \
seed=1 \
+device=cuda:0 > nohup_catching_and_throwing_opt_reward_grok_28000 2>&1 &





