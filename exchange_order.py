import torch

def g1_to_h1(g1, device='cpu'):
    h1 = torch.zeros(19, dtype=torch.float).to(device)
    # 下肢映射 (左腿)
    h1[0] = g1[2]   # left_hip_yaw_joint
    h1[1] = g1[1]   # left_hip_roll_joint
    h1[2] = g1[0]   # left_hip_pitch_joint
    h1[3] = g1[3]   # left_knee_joint
    h1[4] = g1[4]   # left_ankle_joint (使用pitch关节)
    
    # 下肢映射 (右腿)
    h1[5] = g1[8]   # right_hip_yaw_joint
    h1[6] = g1[7]   # right_hip_roll_joint
    h1[7] = g1[6]   # right_hip_pitch_joint
    h1[8] = g1[9]   # right_knee_joint
    h1[9] = g1[10]  # right_ankle_joint (使用pitch关节)
    
    # 躯干映射
    h1[10] = g1[12] # torso_joint (原waist_yaw_joint)
    
    # 上肢映射 (左臂)
    h1[11] = g1[13] # left_shoulder_pitch_joint
    h1[12] = g1[14] # left_shoulder_roll_joint
    h1[13] = g1[15] # left_shoulder_yaw_joint
    h1[14] = g1[16] # left_elbow_joint
    
    # 上肢映射 (右臂)
    h1[15] = g1[18] # right_shoulder_pitch_joint
    h1[16] = g1[19] # right_shoulder_roll_joint
    h1[17] = g1[20] # right_shoulder_yaw_joint
    h1[18] = g1[21] # right_elbow_joint
    
    return h1

def batch_h1_to_g1(h1_batch):
    """
    批量将h1_19dof动作转换为g1_23dof动作（向量化版本）
    输入: [batch_size, 19] 的h1动作张量
    输出: [batch_size, 23] 的g1动作张量
    """
    device = h1_batch.device
    batch_size = h1_batch.shape[0]
    
    # 创建初始化为0的g1动作张量
    g1_batch = torch.zeros(batch_size, 23, device=device)
    
    # 定义源索引和目标索引的映射关系
    src_indices = [2, 1, 0, 3, 4, 7, 6, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    dst_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21]
    
    # 一次性完成所有数据复制
    g1_batch[:, dst_indices] = h1_batch[:, src_indices]
    
    return g1_batch

# def h1_to_g1(h1):
#     """
#     将h1_19dof模型的关节顺序转换为g1_23dof模型的关节顺序
#     输入: 长度为19的h1动作列表 (h1_19dof.xml中的actuator顺序)
#     输出: 长度为23的g1动作列表 (g1_23dof.xml中的actuator顺序)，缺失关节补0
#     """
#     # 创建初始化为0的g1动作列表
#     g1 = torch.zeros(23, dtype=torch.float)
    
#     # 下肢映射 (左腿)
#     g1[0] = h1[2]   # left_hip_pitch_joint
#     g1[1] = h1[1]   # left_hip_roll_joint
#     g1[2] = h1[0]   # left_hip_yaw_joint
#     g1[3] = h1[3]   # left_knee_joint
#     g1[4] = h1[4]   # left_ankle_pitch_joint
#     # left_ankle_roll_joint (索引5) 无对应，保持0
    
#     # 下肢映射 (右腿)
#     g1[6] = h1[7]   # right_hip_pitch_joint
#     g1[7] = h1[6]   # right_hip_roll_joint
#     g1[8] = h1[5]   # right_hip_yaw_joint
#     g1[9] = h1[8]   # right_knee_joint
#     g1[10] = h1[9]  # right_ankle_pitch_joint
#     # right_ankle_roll_joint (索引11) 无对应，保持0
    
#     # 躯干映射
#     g1[12] = h1[10] # waist_yaw_joint (原torso_joint)
    
#     # 上肢映射 (左臂)
#     g1[13] = h1[11] # left_shoulder_pitch_joint
#     g1[14] = h1[12] # left_shoulder_roll_joint
#     g1[15] = h1[13] # left_shoulder_yaw_joint
#     g1[16] = h1[14] # left_elbow_joint
#     # left_wrist_roll_joint (索引17) 无对应，保持0
    
#     # 上肢映射 (右臂)
#     g1[18] = h1[15] # right_shoulder_pitch_joint
#     g1[19] = h1[16] # right_shoulder_roll_joint
#     g1[20] = h1[17] # right_shoulder_yaw_joint
#     g1[21] = h1[18] # right_elbow_joint
#     # right_wrist_roll_joint (索引22) 无对应，保持0
    
#     return g1  


def history_g1_to_h1(g1, device='cpu'):
    return torch.cat([g1_to_h1(g1[i*23:(i+1)*23]) for i in range(4)], dim=0).to(device)
    

def history_unpack_combine(history, device='cpu'):
    action_buf = history[:92]
    ang_vel_buf = history[92:104]
    dof_pos_buf = history[104:196]
    dof_vel_buf = history[196:288]
    proj_g_buf = history[288:300]
    ref_motion_phase_buf = history[300:304]

    action_h1_buf = history_g1_to_h1(action_buf, device)
    ang_vel_h1_buf = ang_vel_buf
    dof_pos_h1_buf = history_g1_to_h1(dof_pos_buf, device)
    dof_vel_h1_buf = history_g1_to_h1(dof_vel_buf, device)
    proj_g_h1_buf = proj_g_buf
    ref_motion_phase_h1_buf = ref_motion_phase_buf
    return torch.cat([action_h1_buf, ang_vel_h1_buf, dof_pos_h1_buf, dof_vel_h1_buf, proj_g_h1_buf, ref_motion_phase_h1_buf], dim=0).to(device)

def actor_unpack_combine(actor_obs, device='cpu'):
    actions = actor_obs[:23]
    base_ang_vel = actor_obs[23:26]
    dof_pos = actor_obs[26:49]
    dof_vel = actor_obs[49:72]
    history_obs_buf = actor_obs[72:376]
    projected_gravity = actor_obs[376:379]
    ref_motion_phase = actor_obs[379:]

    actions_h1 = g1_to_h1(actions, device)
    base_ang_vel_h1 = base_ang_vel
    dof_pos_h1 = g1_to_h1(dof_pos, device)
    dof_vel_h1 = g1_to_h1(dof_vel, device)
    history_h1 = history_unpack_combine(history_obs_buf, device)
    projected_gravity_h1 = projected_gravity
    ref_motion_phase_h1 = ref_motion_phase
    
    return torch.cat([actions_h1, base_ang_vel_h1, dof_pos_h1, dof_vel_h1, history_h1, projected_gravity_h1, ref_motion_phase_h1], dim=0).to(device)

g1_to_h1_joint = torch.tensor([
        2, 1, 0, 3, 4,     # 左腿
        8, 7, 6, 9, 10,    # 右腿
        12, 13, 14, 15, 16, # 左臂
        18, 19, 20, 21      # 右臂
    ], dtype=torch.long, device="cuda:0")

def vectorized_g1_to_h1(g1_batch, index_tensor=g1_to_h1_joint):
    """ 批量转换g1到h1格式 """
    return g1_batch[:, index_tensor]

def vectorized_history_g1_to_h1(target_batch, g1_index):
    return torch.cat([vectorized_g1_to_h1(target_batch[:, i*23:(i+1)*23], g1_index) for i in range(4)], dim=1)

def vectorized_history_unpack(history_batch, g1_index, device='cpu'):
    """
    批量处理历史数据，保持原始顺序:
    [action_h1_buf, ang_vel_buf, dof_pos_h1_buf, dof_vel_h1_buf, proj_g_buf, ref_motion_phase_buf]
    """
    batch_size = history_batch.shape[0]
    
    # 提取各部分数据 (保持原始顺序)
    action_buf = history_batch[:, :92]       # 0-92
    ang_vel_buf = history_batch[:, 92:104]    # 92-104
    dof_pos_buf = history_batch[:, 104:196]   # 104-196
    dof_vel_buf = history_batch[:, 196:288]   # 196-288
    proj_g_buf = history_batch[:, 288:300]    # 288-300
    ref_motion_phase_buf = history_batch[:, 300:304]  # 300-304
    
    
    # 向量化转换
    action_h1_buf = vectorized_history_g1_to_h1(action_buf, g1_index)
    dof_pos_h1_buf = vectorized_history_g1_to_h1(dof_pos_buf, g1_index)
    dof_vel_h1_buf = vectorized_history_g1_to_h1(dof_vel_buf, g1_index)
    
    # 按原始顺序拼接所有部分
    return torch.cat([
        action_h1_buf,        # 76维 (19×4)
        ang_vel_buf,          # 12维
        dof_pos_h1_buf,       # 76维
        dof_vel_h1_buf,       # 76维
        proj_g_buf,           # 12维
        ref_motion_phase_buf  # 4维
    ], dim=1).to(device)

def vectorized_actor_unpack(actor_obs_batch, g1_index, device='cpu'):
    """ 批量处理actor观察值 """
    # 分割各部分数据
    actions = actor_obs_batch[:, :23]
    base_ang_vel = actor_obs_batch[:, 23:26]
    dof_pos = actor_obs_batch[:, 26:49]
    dof_vel = actor_obs_batch[:, 49:72]
    history = actor_obs_batch[:, 72:376]
    proj_gravity = actor_obs_batch[:, 376:379]
    motion_phase = actor_obs_batch[:, 379:]

    # 向量化转换
    actions_h1 = vectorized_g1_to_h1(actions, g1_index)
    dof_pos_h1 = vectorized_g1_to_h1(dof_pos, g1_index)
    dof_vel_h1 = vectorized_g1_to_h1(dof_vel, g1_index)

    history_h1 = vectorized_history_unpack(history, g1_index, device)
    
    return torch.cat([
        actions_h1, 
        base_ang_vel,
        dof_pos_h1,
        dof_vel_h1,
        history_h1,
        proj_gravity,
        motion_phase
    ], dim=1).to(device)


g1_to_h1_body = torch.tensor([
        0,    # 0: pelvis -> 0
        3,    # 1: left_hip_pitch_link -> 3
        2,    # 2: left_hip_roll_link -> 2
        1,    # 3: left_hip_yaw_link -> 1
        4,    # 4: left_knee_link -> 4
        5,    # 5: left_ankle_pitch_link -> 5 (H1的left_ankle_link)
        8,    # 7: right_hip_pitch_link -> 8
        7,    # 8: right_hip_roll_link -> 7
        6,    # 9: right_hip_yaw_link -> 6
        9,    # 10: right_knee_link -> 9
        10,   # 11: right_ankle_pitch_link -> 10 (H1的right_ankle_link)
        10,   # 13: torso_link -> 10
        11,   # 14: left_shoulder_pitch_link -> 11
        12,   # 15: left_shoulder_roll_link -> 12
        13,   # 16: left_shoulder_yaw_link -> 13
        14,   # 17: left_elbow_link -> 14
        15,   # 19: right_shoulder_pitch_link -> 15
        16,   # 20: right_shoulder_roll_link -> 16
        17,   # 21: right_shoulder_yaw_link -> 17
        18,   # 22: right_elbow_link -> 18
    ], dtype=torch.long)

g1_to_h1_joint = torch.tensor([
        2, 1, 0, 3, 4,     # 左腿
        8, 7, 6, 9, 10,    # 右腿
        12, 13, 14, 15, 16, # 左臂
        18, 19, 20, 21      # 右臂
    ], dtype=torch.long)

def vectorized_history_critic_unpack(history_batch, g1_index, device='cpu'):
    actions = history_batch[:, :92]
    base_ang_vel = history_batch[:, 92:104]
    base_lin_vel = history_batch[:, 104:116]
    dof_pos = history_batch[:, 116:208]
    dof_vel = history_batch[:, 208:300]
    projected_gravity = history_batch[:, 300:312]
    ref_motion_phase = history_batch[:, 312:316]

    actions_h1 = vectorized_history_g1_to_h1(actions, g1_index)
    dof_pos_h1 = vectorized_history_g1_to_h1(dof_pos, g1_index)
    dof_vel_h1 = vectorized_history_g1_to_h1(dof_vel, g1_index)

    return torch.cat([
        actions_h1,
        base_ang_vel,
        base_lin_vel,
        dof_pos_h1,
        dof_vel_h1,
        projected_gravity,
        ref_motion_phase
    ], dim=1).to(device)

def vectorized_body_unpack(body_pos_batch, g1_body, device='cpu'):
    batch_num = body_pos_batch.shape[0]
    body_batch = body_pos_batch[:, :72]
    body_extend_batch = body_pos_batch[:, 72:81]

    body_batch = body_batch.reshape(batch_num, 24, 3)
    body_batch_h1 = vectorized_g1_to_h1(body_batch, g1_body)
    body_batch_h1 = body_batch_h1.reshape(batch_num, 60)
    
    return torch.cat([
        body_batch_h1,
        body_extend_batch
    ], dim=1).to(device)

def vectorized_critic_unpack(critic_obs_batch, device='cpu'):
    g1_joint = g1_to_h1_joint.to(device)
    g1_body = g1_to_h1_body.to(device)

    actions = critic_obs_batch[:, :23]
    base_ang_vel = critic_obs_batch[:, 23:26]
    base_lin_vel = critic_obs_batch[:, 26:29]
    dif_local_rigid_body_pos = critic_obs_batch[:, 29:110]
    dof_pos = critic_obs_batch[:, 110:133]
    dof_vel = critic_obs_batch[:, 133:156]
    dr_base_com = critic_obs_batch[:, 156:159]
    dr_ctrl_delay = critic_obs_batch[:, 159:160]
    dr_friction = critic_obs_batch[:, 160:161]
    dr_kd = critic_obs_batch[:, 161:184]
    dr_kp = critic_obs_batch[:, 184:207]
    dr_link_mass = critic_obs_batch[:, 207:229]
    history_critic = critic_obs_batch[:, 229:545]
    local_ref_rigid_body_pos = critic_obs_batch[:, 545:626]
    projected_gravity = critic_obs_batch[:, 626:629]
    ref_motion_phase = critic_obs_batch[:, 629:630]

    actions_h1 = vectorized_g1_to_h1(actions, g1_joint)
    dif_local_rigid_body_pos_h1 = vectorized_body_unpack(dif_local_rigid_body_pos, g1_body, device)
    dof_pos_h1 = vectorized_g1_to_h1(dof_pos, g1_joint)
    dof_vel_h1 = vectorized_g1_to_h1(dof_vel, g1_joint)
    dr_kd_h1 = vectorized_g1_to_h1(dr_kd, g1_joint)
    dr_kp_h1 = vectorized_g1_to_h1(dr_kp, g1_joint)
    history_critic_h1 = vectorized_history_critic_unpack(history_critic, g1_joint, device)
    local_ref_rigid_body_pos_h1 = vectorized_body_unpack(local_ref_rigid_body_pos, g1_body, device)

    return torch.cat([
        actions_h1,
        base_ang_vel,
        base_lin_vel,
        dif_local_rigid_body_pos_h1,
        dof_pos_h1,
        dof_vel_h1,
        dr_base_com,
        dr_ctrl_delay,
        dr_friction,
        dr_kd_h1,
        dr_kp_h1,
        dr_link_mass,
        history_critic_h1,
        local_ref_rigid_body_pos_h1,
        projected_gravity,
        ref_motion_phase
    ], dim=1).to(device)