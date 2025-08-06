nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=main_all_friction_JointSame_overturn \
+rewards=motion_tracking/ \
experiment_name=_opt_DR_grok \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/" \
seed=1 \
+device=cuda:0 > nohup__opt_DR_grok 2>&1 &


[32.26]
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=domainrand_grok_FrontKick \
+rewards=motion_tracking/main_grok20000_opt_front_kick_ExtraMotionDesc \
experiment_name=FrontKick_opt_DR_grok \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/0026_kicking1_stageii_RootLifted_0.04.pkl" \
seed=1 \
+device=cuda:0 > nohup_FrontKick_opt_DR_grok 2>&1 &


[111] 8.6 / 10:21
nohup python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof \
+domain_rand=domainrand_grok_FrontKick \
+rewards=motion_tracking/main_grok_OptReward_OptDR_OptReward_FrontKick \
experiment_name=FrontKick_opt_DR_grok_2optReward \
robot.motion.motion_file="smpl_retarget/retargeted_motion_data/mink/0026_kicking1_stageii_RootLifted_0.04.pkl" \
seed=1 \
+device=cuda:0 > nohup_FrontKick_opt_DR_grok_2optReward 2>&1 &