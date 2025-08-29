pip install isaacsim[all]==4.5.0 --extra-index-url https://pypi.nvidia.com

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install mujoco

the python version must be satisfied:python == 3.10






####----------------How to run-------------------####



1,When you first run this code, go to the Config.py change the vairable "file_path", it points to the 3D model. SF_TRON1A/USD/TRON.usd.
It should be absolute path of the TRON.usd in your computer.
Example:
file_path = "C:/Users/21363/PycharmProjects/Isaac_Lab/TRON/SF_TRON_continuous_Vcmd/SF_TRON1A/USD/TRON.usd"  must be absolute path


2,in the beginning of PPO_Isaac_continuous_Vcmd.py, there is a variable called headless. 1 means no UI, 0 means show UI. 

3, Run Train.py to train, Run Play.py or Sim2Sim.py to play
