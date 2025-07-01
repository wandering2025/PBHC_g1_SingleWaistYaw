import pickle
import os
import argparse
import numpy as np

# 指定你的PKL文件路径
#file_path = '/home/bbw/PBHC_g1_SingleWaistYaw/smpl_retarget/retargeted_motion_data/mink/hips_poses.pkl'
clipping_threshold = 0.0004

def load_pkl_as_dict(filepath):
    """
    加载一个PKL文件并将其内容作为Python字典返回。

    参数:
    filepath (str): PKL文件的完整路径。

    返回:
    dict: 从PKL文件加载的数据，如果文件内容不是字典，则返回None。
          如果发生错误，则返回None并打印错误信息。
    """
    if not os.path.exists(filepath):
        print(f"错误: 文件 '{filepath}' 未找到。请检查文件路径是否正确。")
        return None
        
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            print(f"loaded '{filepath}' successfully")
            return data
        else:
            print(f"警告: 文件 '{filepath}' 中的内容不是一个字典类型。")
            return None
    except pickle.UnpicklingError as e:
        print(f"错误: 无法解压文件 '{filepath}'。可能文件已损坏或不是有效的pickle文件。详细信息: {e}")
        return None
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None
    
def save_dict_as_pkl(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"成功保存新的PKL文件到: {filepath}")



def main():

    # 调用函数加载数据
    loaded_file = load_pkl_as_dict(file_path)

    first_level_key = list(loaded_file.keys())[0] 
    inner_dict = loaded_file[first_level_key]
    # for inner_key in inner_dict.keys():
    #     print(inner_key)
    dof_content = inner_dict['dof']
    print(dof_content.shape)
    
    for i in range(len(dof_content)):
        # print(f"frame{i} left_ankle_pitch_joint:{dof_content[i][4]} left_ankle_roll_joint:{dof_content[i][5]}")
        # print(f"frame{i} right_ankle_pitch_joint:{dof_content[i][10]} right_ankle_roll_joint:{dof_content[i][11]}")
        
        left_ankle_pitch_joint = dof_content[i][4]
        left_ankle_roll_joint = dof_content[i][5]
        right_ankle_pitch_joint = dof_content[i][10]
        right_ankle_roll_joint = dof_content[i][11]
            # Using a ternary operator for concise clipping
        
        clipped_left_ankle_pitch = max(-clipping_threshold, min(left_ankle_pitch_joint, clipping_threshold))
        clipped_left_ankle_roll = max(-clipping_threshold, min(left_ankle_roll_joint, clipping_threshold))
        clipped_right_ankle_pitch = max(-clipping_threshold, min(right_ankle_pitch_joint, clipping_threshold))
        clipped_right_ankle_roll = max(-clipping_threshold, min(right_ankle_roll_joint, clipping_threshold))

        dof_content[i][4] = clipped_left_ankle_pitch
        dof_content[i][5] = clipped_left_ankle_roll
        dof_content[i][10] = clipped_right_ankle_pitch
        dof_content[i][11] = clipped_right_ankle_roll


        # print('--------------clipped------------------')
        # print(f"frame{i} left_ankle_pitch_joint:{dof_content[i][4]} left_ankle_roll_joint:{dof_content[i][5]}")
        # print(f"frame{i} right_ankle_pitch_joint:{dof_content[i][10]} right_ankle_roll_joint:{dof_content[i][11]}")
        # print('#################################################################')

    dir_name = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    new_file_name = f"{base_name}_AnkleClipped_{clipping_threshold}.pkl"
    new_file_path = os.path.join(dir_name, new_file_name)

    save_dict_as_pkl(loaded_file, new_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="加载PKL文件，提升root_trans_offset的Z轴，并保存新文件。")
    parser.add_argument('--pkl_path', type=str, required=True,
                        help="--pkl_path path_to_your_.pkl")
    
    # 解析命令行参数
    args = parser.parse_args()
    file_path = args.pkl_path
    main()
