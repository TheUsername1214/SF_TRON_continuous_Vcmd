import numpy as np

joint_names = ['abad_L_Joint',
               'abad_R_Joint',
               'hip_L_Joint',
               'hip_R_Joint',
               'knee_L_Joint',
               'knee_R_Joint',
               'ankle_L_Joint',
               'ankle_R_Joint'
               ]
def get_actuator_joint_angles(model, data):
    """获取所有执行器对应的关节角度"""
    global joint_names

    joint_angles = []
    for joint_name in joint_names:
        joint_id = model.joint(joint_name).id
        joint_qpos_addr = model.jnt_qposadr[joint_id]
        joint_angles.append(data.qpos[joint_qpos_addr])

    return np.array(joint_angles)


def get_actuator_joint_velocities(model, data):
    """获取所有执行器对应的关节角速度"""
    global joint_names

    joint_velocities = []
    for joint_name in joint_names:
        joint_id = model.joint(joint_name).id
        joint_qvel_addr = model.jnt_dofadr[joint_id]  # 注意这里用的是 jnt_dofadr
        joint_velocities.append(data.qvel[joint_qvel_addr])

    return np.array(joint_velocities)