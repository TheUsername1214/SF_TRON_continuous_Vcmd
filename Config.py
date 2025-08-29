class RobotConfig:
    class CriticParam:  # Critic 神经网络 参数
        state_dim = 24
        privilege_dim = 8
        critic_layers_num = 256
        critic_lr = 1e-4
        critic_update_frequency = 150

    class ActorParam:  # Actor 神经网络 参数
        act_layers_num = 256
        actuator_num = 8
        actor_lr = 1e-4
        actor_update_frequency = 20

    class PPOParam:  # 强化学习 PPO算法 参数
        gamma = 0.99
        lam = 0.95
        epsilon = 0.2
        maximum_step = 30
        episode = 1000
        entropy_coef = -1  # positive means std increase, else decrease
        batch_size = 25000

    class EnvParam:  # 训练环境的参数
        agents_num = 1500
        agents_num_in_play = 1
        file_path = "C:/Users/21363/PycharmProjects/Isaac_Lab/TRON/SF_TRON_continuous_Vcmd/SF_TRON1A/USD/TRON.usd"  # abs path, not relative path
        prim_path = "SF_TRON"  # Relative to /World`
        dt = 0.02
        sub_step = 10
        friction_coef = 1

    class RobotParam:  # 机器人的参数
        action_scale = 1
        std_scale = 1
        Kp = 45
        Kd = 3.5
        initial_height = 0.85
        initial_body_vel_range = 0.1
        initial_joint_pos_range = 0.1
        initial_joint_vel_range = 0.1
        initial_target_angle = [0,0,
                                -0,0,
                                -1,1,
                                0,0]

    class MissionObjectiveParam:  # 训练目标
        target_vel_scale = -1.5
        target_height = 1

    class Gait:
        gait_frequency = 1
        swing_height = [0.1, 0.2]

    class DomainRandomization:
        com_range = [0.8, 1.2]
        mass_range = [0.8, 1.2]
        inertia_range = [0.8, 1.2]
        friction_range = [0.5, 1.5]
        Kd_range = [0.8, 1.2]
        Kp_range = [0.8, 1.2]

    class Normalization:
        joint_angle_normalized_factor = 1
        joint_vel_normalized_factor = 1

    class ObsNoise:
        joint_angle_obs_noise_range = 0.1
        joint_vel_obs_noise_range = 0.1
        body_ori_obs_noise_range = 0.1
