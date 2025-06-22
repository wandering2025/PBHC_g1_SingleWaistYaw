import numpy as np
import sys
import os
from scipy.spatial.transform import Rotation as sRot

# 定义坐标系转换矩阵
R_cam_to_world = np.array([[1, 0, 0],
                           [0, 0, 1],
                           [0, -1, 0]])

# 检查命令行参数
if len(sys.argv) != 2:
    print("Usage: python npy2npz_convert.py <video_name>")
    sys.exit(1)

video_name = sys.argv[1]
tram_file_path = "/home/bbw/PBHC_g1_SingleWaistYaw/tram_out_npy/MU-CR7-solo_siuuu.npy"

# 检查文件是否存在
if not os.path.exists(tram_file_path):
    print(f"Error: File {tram_file_path} does not exist.")
    sys.exit(1)

# 加载数据
track_data = np.load(tram_file_path, allow_pickle=True).item()

# 检查必需的键
required_keys = ['pred_rotmat', 'pred_trans', 'pred_shape', 'pred_cam']
for key in required_keys:
    if key not in track_data:
        print(f"Error: Missing key '{key}' in {tram_file_path}")
        sys.exit(1)

# 转换为 numpy 数组
pred_rotmat = track_data["pred_rotmat"].numpy()
trans = track_data["pred_trans"].numpy().squeeze(1)  # [N, 3]
trans_world = (R_cam_to_world @ trans.T).T  # 转换为世界坐标系
shape = track_data["pred_shape"].numpy()
duration = track_data['pred_cam'].shape[0]

# 定义根关节校正旋转
R_correct = sRot.from_euler('x', -90, degrees=True).as_matrix()

# 转换姿态数据
poses = np.zeros((duration, 72))
for i in range(duration):
    for j in range(24):
        if j == 0:  # 根关节
            # print(f"Frame {i} - 原始根关节旋转矩阵:\n{pred_rotmat[i, 0]}")  # 打印原始矩阵
            R_root_corrected = R_correct @ pred_rotmat[i, 0]
            # print(f"Frame {i} - 校正后的根关节旋转矩阵:\n{R_root_corrected}")  # 打印校正后矩阵
            axis_angle = sRot.from_matrix(R_root_corrected).as_rotvec()
            # print(f"Frame {i} - 校正后的轴角表示: {axis_angle}")  # 打印轴角表示
            poses[i, :3] = axis_angle
        else:  # 其他关节
            axis_angle = sRot.from_matrix(pred_rotmat[i, j]).as_rotvec()
            poses[i, 3 + 3*(j-1) : 3 + 3*(j-1) + 3] = axis_angle

# 处理平移和形状
N = duration
# trans = trans.squeeze(1)
betas = shape.mean(axis=0)  # 使用 numpy 的 mean

# 填充缺失数据
gender = "neutral"
mocap_framerate = 30
dmpls = np.zeros((N, 8))

# 确保输出目录存在
output_dir = "/home/bbw/PBHC_g1_SingleWaistYaw/smpl_retarget/motion_debug/selected"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "MU-CR7-solo_siuuu_transfered.npz")

# 保存数据
np.savez_compressed(
    output_path,
    trans=trans_world,
    gender=gender,
    mocap_framerate=mocap_framerate,
    betas=betas,
    dmpls=dmpls,
    poses=poses
)