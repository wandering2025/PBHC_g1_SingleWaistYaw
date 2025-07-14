import numpy as np

f_v = 0.01
dof_vel = np.random.uniform(low=0, high=1, size=(23,))

f_c = 0.1
r = 35e-3 
F_trans = np.random.uniform(low=0, high=1, size=(23,)) 

J = np.random.uniform(low=0, high=1, size=(23,))
dof_ang_acc = np.random.uniform(low=0, high=1, size=(23,))

torque_a = np.random.uniform(low=0, high=5, size=(23,))

tao_load =np.random.uniform(low=0, high=1, size=(23,))


def sgn(x):
    return 1 if x >= 0 else -1

def predict_dof_vel(f_c,r,F_trans,J,dof_ang_acc,torque_a,tao_load, f_v, dof_vel):
    dof_vel_pre = 0.0
    dof_vel_pre = (-f_c * r * F_trans * sgn(dof_vel) - J * dof_ang_acc + torque_a + tao_load ) / f_v

    return dof_vel_pre

def main():
    
    for i in range(len(dof_vel)):
        
        dof_vel_j = dof_vel[i]
        F_trans_j = F_trans[i]
        J_j = J[i]
        dof_ang_acc_j = dof_ang_acc[i]
        torque_a_j = torque_a[i]
        tao_load_j = tao_load[i]

        print(f'joint {i}')
        dof_vel_pre = predict_dof_vel(f_c, r, F_trans_j, J_j, dof_ang_acc_j, torque_a_j, tao_load_j, f_v, dof_vel_j)
        print(f'predicted dof_vel: {dof_vel_pre}')



if __name__ == "__main__":
    main()