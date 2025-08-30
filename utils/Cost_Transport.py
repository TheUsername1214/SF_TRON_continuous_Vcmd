from matplotlib.pyplot import *
import numpy as np


class Cost_Transport():
    def __init__(self):
        self.joint_angle = []
        self.joint_vel = []
        self.joint_torque = []
        self.motor_power = []
        self.total_energy = []
        self.time = []
        self.average_vel = []

        self.motor_energy = 0
        self.body_energy = 0
        self.name = ['abad_L_Joint', 'abad_R_Joint', 'hip_L_Joint', 'hip_R_Joint', 'knee_L_Joint', 'knee_R_Joint',
                     'ankle_L_Joint', 'ankle_R_Joint']

    def compute_motor_energy(self, total_energy, torque, angle, angular_vel, time, dt):
        self.motor_energy += np.sum(np.clip(torque * angular_vel * dt, 0, 1111))
        self.motor_power.append(np.clip(torque * angular_vel, 0, 1111))
        self.joint_torque.append(torque)
        self.joint_angle.append(angle)
        self.joint_vel.append(angular_vel)
        self.total_energy.append([total_energy])
        self.time.append(time)


    def compute_body_energy(self, body_xy_pos, body_mass):
        self.body_energy = body_xy_pos * body_mass * 9.81
        print(body_xy_pos/self.time[-1][0])

    def plot_each_angle(self):
        figure()
        for i in range(8):
            plot(np.array(self.time).flatten(),
                 np.array(self.joint_angle).flatten())
        xlabel("time[sec]")
        ylabel("angle[rad]")
        title("joint angle")

    def plot_each_vel(self):
        figure()
        for i in range(8):
            subplot(4, 2, i + 1)
            plot(np.array(self.time).flatten(),
                 np.array(self.joint_vel)[:, i].flatten())
            title(f"{self.name[i]}")
            ylim(-10, 10)
        suptitle("angular vel")

    def plot_each_torque(self):
        figure()
        for i in range(8):
            subplot(4, 2, i + 1)
            plot(np.array(self.time).flatten(),
                 np.array(self.joint_torque)[:, i].flatten(), label=f"{self.name[i]}")
            title(f"{self.name[i]}")
            ylim(0,45)
        legend()
        suptitle("Torque")

    def plot_each_power(self):
        figure()

        for i in range(8):
            subplot(4, 2, i + 1)
            plot(np.array(self.time).flatten(),
                 np.array(self.motor_power)[:, i].flatten(), label=f"{self.name[i]}")
            title(f"{self.name[i]}")
            ylim(0, 100)
        legend()
        suptitle("Power")

    def plot_energy(self):
        figure()

        plot(np.array(self.time),
             np.array(self.total_energy)[:, 0], label=f"energy")
        legend()
        xlabel("time[sec]")
        ylabel("total energy[W]")
        title("total energy")
        show()

    def reset(self):
        self.motor_energy = 0
        self.body_energy = 0

    def get_cot(self):
        return self.motor_energy / self.body_energy
