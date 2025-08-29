import mujoco.viewer

from Config import *
from utils.NN import *
from utils.Useful_Function import *
from utils.Mujoco_Utils import *
from utils.Cost_Transport import *

COT = Cost_Transport()
state_dim = RobotConfig.CriticParam.state_dim
act_layers_num = RobotConfig.ActorParam.act_layers_num
actuator_num = RobotConfig.ActorParam.actuator_num

substep = RobotConfig.EnvParam.sub_step
dt = RobotConfig.EnvParam.dt
Kp = RobotConfig.RobotParam.Kp
Kd = RobotConfig.RobotParam.Kd
action_scale = RobotConfig.RobotParam.action_scale
initial_angle = FT(RobotConfig.RobotParam.initial_target_angle)
mujoco_dt = 0.001

device = torch.device("cuda:0")
actor = Actor(state_dim, act_layers_num, actuator_num,action_scale=action_scale).to(device)
actor.load_state_dict(torch.load('model/actor.pth'))
model = mujoco.MjModel.from_xml_path("SF_TRON1A/urdf/robot.xml")
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

v_cmd = FT([-.5, 0, 0])
sim_time = FT([0])

with mujoco.viewer.launch_passive(model, data) as viewer:
    i = 0
    while True:
        sim_time += mujoco_dt
        # 读取传感器数据 (包括IMU数据)
        joint_angles = FT(get_actuator_joint_angles(model, data))  # 所有关节位置
        joint_velocities = FT(get_actuator_joint_velocities(model, data))  # 所有关节速度
        sensor_data1 = data.qpos[3:7]
        sensor_data2 = data.qvel[3:6]
        euler_angle = get_euler_angle(FT([sensor_data1])).flatten()
        angular_vel = FT(sensor_data2)

        if torch.any(torch.abs(euler_angle) > np.pi / 3) or i > 20000:
            i = 0
            body_pos = np.sum(np.sqrt((data.qpos[:2]-start_pos) ** 2))

            COT.compute_body_energy(body_pos, 20.81)
            mujoco.mj_resetData(model, data)
            cot = COT.get_cot()
            break
        if (i % int(dt / mujoco_dt)) == 0:
            sine_clock = torch.sin(2 * torch.pi * sim_time)
            cosine_clock = torch.cos(2 * torch.pi * sim_time)
            state = torch.concatenate((joint_angles,
                                       joint_velocities,
                                       euler_angle,
                                       v_cmd,
                                       sine_clock,
                                       cosine_clock
                                       )).reshape(1, -1)

            angle, _ = actor(state)


            viewer.sync()
        i += 1
        torque = Kp * (angle[0]+initial_angle - joint_angles) - Kd * joint_velocities
        torque[:6] = torque[:6].clip(-80, 80)
        torque[-2:] = torque[-2:].clip(-20, 20)
        data.ctrl[:] = torque.detach().cpu().numpy()

        mujoco.mj_step(model, data)

        joint_angles = FT(get_actuator_joint_angles(model, data))  # 所有关节位置
        joint_velocities = get_actuator_joint_velocities(model, data)  # 所有关节速度

        if i > 500:

            if i == 501:
                start_pos = data.qpos[:2].copy()
            COT.compute_motor_energy(data.energy[0] + data.energy[1], torque.detach().cpu().numpy(), joint_angles,
                                     joint_velocities,
                                     sim_time.cpu().numpy(), 0.001)

print("cost: " ,cot)
COT.plot_each_torque()
COT.plot_each_power()
COT.plot_each_vel()
COT.plot_energy()
