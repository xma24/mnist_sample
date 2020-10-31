import os

port_number = 10001

os.system(
    "cd " + folder1 + ";" + "ls" + "; /usr/bin/screen /home/xma24/anaconda3/envs/pytorch/bin/tensorboard --logdir ./lightning_logs/ --port " + str(
        port_number))
