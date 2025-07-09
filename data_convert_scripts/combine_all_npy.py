import numpy as np
import os
import argparse


def combine_npy_files_with_text_motions(npy_folder_path, output_filename_base):
    """
    合并指定文件夹中所有具有相同结构的 .npy 文件。
    'motions' 将作为文本字符串列表保留，'dof_pos' 也将作为数组列表保留。

    Args:
        npy_folder_path (str): 包含所有 .npy 文件的文件夹路径。
        output_filename (str): 合并后的 .npy 文件名。
    """
    #dict_keys(['link_angular_acceleration', 'base_angular_vel', 'projected_gravity', 'dof_pos', 'dof_vel', 'torque'])
    all_link_angular_list = [] 
    all_base_angular_vel_list = []
    all_projected_gravity_list = [] 
    all_dof_pos_list = []  
    all_dof_vel_list = [] 
    all_torque_list = []  
    

    # 获取所有 .npy 文件的路径，并确保按某种顺序处理（例如，按文件名排序）
    npy_files = sorted([f for f in os.listdir(npy_folder_path) if f.endswith('.npy')])
    npy_count = len(npy_files)

    if not npy_files:
        print(f"在 {npy_folder_path} 中没有找到 .npy 文件。")
        return

    print(f"检测到 {len(npy_files)} 个 .npy 文件，开始合并...")

    for i, filename in enumerate(npy_files):
        
        npy_path = os.path.join(npy_folder_path, filename)
        try:
            saved_data = np.load(npy_path, allow_pickle=True).item()

            # 验证数据结构
            if not isinstance(saved_data, dict) or 'link_angular_acceleration' not in saved_data \
            or 'base_angular_vel' not in saved_data or 'projected_gravity' not in saved_data \
            or 'dof_pos' not in saved_data or 'dof_vel' not in saved_data or 'torque' not in saved_data:
                
                print(f"警告：文件 {filename} 的结构不符合预期，跳过。")
                continue



            # 提取数据
            all_link_angular_list.append(saved_data.get('link_angular_acceleration', None))  # 将 link_angular_acceleration 数组添加到列表中
            all_base_angular_vel_list.append(saved_data.get('base_angular_vel', None))  # 将 base_angular_vel 数组添加到列表中
            all_projected_gravity_list.append(saved_data.get('projected_gravity', None))  # 将 projected_gravity 数组添加到列表中
            all_dof_pos_list.append(saved_data['dof_pos'])     # 将独立的 dof_pos 数组添加到列表中
            all_dof_vel_list.append(saved_data.get('dof_vel', None))  # 将 dof_vel 数组添加到列表中
            all_torque_list.append(saved_data.get('torque', None    ))  # 将 torque 数组添加到列表中


            if (i + 1) % 100 == 0 or (i + 1) == len(npy_files):
                print(f"已处理 {i + 1}/{len(npy_files)} 个文件。")


        except Exception as e:
            print(f"加载或处理文件 {filename} 时发生错误：{e}")
            continue

    if not all_link_angular_list or not all_base_angular_vel_list or not all_dof_pos_list or not all_torque_list or not all_projected_gravity_list or not all_dof_vel_list:
        # 如果没有任何数据可合并，打印提示信息并返回
        print("没有可合并的数据。")
        return

    # 创建新的字典
    # 'motions' 和 'dof_pos' 的值现在都是列表
    combined_data = {
        'link_angular_acceleration': all_link_angular_list,  # 这里是一个包含所有 link_angular_acceleration 数组的列表
        'base_angular_vel': all_base_angular_vel_list,      # 这里是一个包含所有 base_angular_vel 数组的列表
        'projected_gravity': all_projected_gravity_list,  # 这里是一个包含所有 projected_gravity 数组的 列表
        'dof_pos': all_dof_pos_list,        # 这里是一个包含所有 dof_pos 数组的列表
        'dof_vel': all_dof_vel_list ,      # 这里是一个包含所有 dof_vel 数组的列表
        'torque': all_torque_list,         # 这里是一个包含所有 torque 数组的列表

    }

    # 保存合并后的数据
    output_filename = f"{output_filename_base}_{npy_count}.npy"
    output_path = os.path.join(npy_folder_path, output_filename)
    np.save(output_path, combined_data, allow_pickle=True)

    print(f"所有文件已成功合并到 {output_path}")
    # print(f"合并后的 'motions' 现在是一个包含 {len(combined_data['motions'])} 个文本字符串的列表。")
    # print(f"合并后的 'dof_pos' 现在是一个包含 {len(combined_data['dof_pos'])} 个 numpy 数组的列表。")
    # print(f"合并后的 'root_rot' 现在是一个包含 {len(combined_data['rpy'])} 个 numpy 数组的列表。")
    # if len(combined_data['motions']) > 0:
    #     print(f"第一个 'motions' 文本示例: '{combined_data['motions'][0]}'")
    # if len(combined_data['dof_pos']) > 0:
    #     print(f"第一个 'dof_pos' 数组的形状: {combined_data['dof_pos'][0].shape}")
    # if len(combined_data['rpy']) > 0:
    #     print(f"第一个 'root_rot' 数组的形状: {combined_data['rpy'][0].shape if combined_data['rpy'][0] is not None else 'None'}")
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="path_to_log_folder")
    parser.add_argument('--log_path', type=str, required=True,
                        help="--log_path path_to_your_.pkl")
    args = parser.parse_args()
    file_path = args.log_path
    count = 0
    combine_npy_files_with_text_motions(npy_folder_path=f'{file_path}/iteration_data_collection',
                                        output_filename_base=f'{file_path}/iteration_data_combined')
    
    #npy_path = '/root/PHC_process/g1_npy_combined.npy'
    # npy_path = '/root/PHC_process/x1_npy_combined.npy'
    # saved_data = np.load(npy_path, allow_pickle=True).item()
    # print(saved_data['motions'][957])
    # print(saved_data['dof_pos'][957].shape)


