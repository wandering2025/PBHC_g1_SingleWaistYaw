import numpy as np

#npy_path = '/root/PBHC_g1_SingleWaistYaw/logs/MotionTracking/20250708_161051-dev_iteration_record-motion_tracking-g1_23dof_lock_wrist/iteration_data_collection/iteration_0_env0_data.npy'
npy_path='/root/PBHC_g1_SingleWaistYaw/logs/MotionTracking/20250708_185026-ForwardLeanWalk__iteration_record-motion_tracking-g1_23dof/iteration_data_combined_35832.npy'

saved_data = np.load(npy_path, allow_pickle=True).item()

print(f"验证加载的数据类型: {type(saved_data)}")

print(f"验证字典键: {saved_data.keys()}")


selected = 'torque'
print(f'selected: {selected}')
print(type(saved_data[selected]))
# print(saved_data[selected].shape)
print(len(saved_data[selected]))

for i in range(len(saved_data[selected])):
    link_angular_acceleration=saved_data['link_angular_acceleration'][i]
    base_angular_vel=saved_data['base_angular_vel'][i]
    projected_gravity=saved_data['projected_gravity'][i]
    dof_pos=saved_data['dof_pos'][i]
    dof_vel=saved_data['dof_vel'][i]
    torque=saved_data['torque'][i]

    if len(link_angular_acceleration) != len(torque) \
    or len(base_angular_vel) != len(torque) \
    or len(projected_gravity) != len(torque) \
    or len(dof_pos) != len(torque) \
    or len(dof_vel) != len(torque):
        print(f"数据长度不匹配: {i}")
        print(f"link_angular_acceleration: {len(link_angular_acceleration)}")
        print(f"base_angular_vel: {len(base_angular_vel)}")
        print(f"projected_gravity: {len(projected_gravity)}")
        print(f"dof_pos: {len(dof_pos)}")
        print(f"dof_vel: {len(dof_vel)}")
        print(f"torque: {len(torque)}")
    else:
        if i == len(saved_data[selected])-1:
            print(f"data shape colleted right")

        

