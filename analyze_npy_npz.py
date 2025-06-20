# coding=utf-8
"""
time:2025/5/11 10:53
file:analyze_tram_npz.py
user:MECHREVO
"""
import numpy as np

import torch

def analyze_npy_file(file_path):
# 加载 hps_track_0.npy
    hps_data = np.load(file_path, allow_pickle=True).item()

    # 打印字典的键
    print("Keys:", hps_data.keys())
    tram_keys = hps_data.keys()
    # 打印每个字段的形状、类型和样本值
    for key, value in hps_data.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}, sample={value[0] if value.shape[0] > 0 else value}")
        else:
            print(f"{key}: type={type(value)}, sample={value[:10] if isinstance(value, list) else value}")


def analyze_npz_file(file_path):
    # 加载 .npz 文件
    npz_data = np.load(file_path, allow_pickle=True)

    # 打印文件中包含的键（文件中所有的数组名）
    print("Keys:", list(npz_data.keys()))
    examp_keys = npz_data.keys()
    #print(examp_keys)
    # 打印每个键对应数组的形状、类型和样本值
    for key in npz_data.keys():
        value = npz_data[key]

        # 如果是 ndarray 类型，进行处理
        if isinstance(value, np.ndarray):
            # 检查是否为零维数组（标量）
            if value.ndim == 0:
                print(f"{key}: shape={value.shape}, dtype={value.dtype}, sample={value}")
            # 检查数组是否为空
            elif value.size > 0:
                print(f"{key}: shape={value.shape}, dtype={value.dtype}, sample={value[0]}")
            else:
                print(f"{key}: shape={value.shape}, dtype={value.dtype}, sample=Empty Array")
        else:
            print(f"{key}: type={type(value)}, sample={value[:10] if isinstance(value, list) else value}")
# main函数
def main():
    
    tram_file_path = r"/home/bbw/PBHC_g1_SingleWaistYaw/description/robots/g1/dof_axis.npy"
    #exmaple_file_path = r"path_to_npz"

    print('---------------npy--------------')
    #hps_data = np.load(tram_file_path, allow_pickle=True).item()
    #print("Keys:", hps_data.keys())
    #analyze_npy_file(tram_file_path)



    #print('------------npz-------------')
    #analyze_npz_file(exmaple_file_path)

    npz_data = np.load(tram_file_path, allow_pickle=True)
    print('dof_axis.npy')
    for i, row in enumerate(npz_data):
        print(f"Row {i:2d}: {row}")


    g1_23_axis_data = [
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],  # This value is different from your original list (was [1 0 0])
    [1, 0, 0],  # This value is different from your original list (was [0 1 0])
    [0, 0, 1],  # This value is different from your original list (was [0 0 1])
    [0, 1, 0],  # This value is different from your original list (was [1 0 0])
    [1, 0, 0],  # This value is different from your original list (was [0 1 0])
    [0, 1, 0],  # This value is different from your original list (was [0 0 1])
    [1, 0, 0],  # This value is different from your original list (was [0 1 0])
    [0, 0, 1],  # This value is different from your original list (was [0 1 0])
    [0, 1, 0],  # This value is different from your original list (was [1 0 0])
    [1, 0, 0]   # This value is different from your original list (was [0 0 1])
    ]
    g1_23_numpy_array = np.array(g1_23_axis_data)
    new_file_path = "description/robots/g1/g1_23dof_dof_axis.npy"
    np.save(new_file_path, g1_23_numpy_array)
    print(f"Successfully created and saved '{new_file_path}'")
    print("Content of the new file:")
    print(g1_23_numpy_array)


# 调用main函数
if __name__ == "__main__":
    main()
