from Runner import Runner
from arguments import get_common_args, get_mixer_args
import random
import numpy as np
from fjspEnvironment import Situation
import torch

MachineNums = [5,10,15,20,25]
JobNums = [10,20,30,40,50]


def writefile(JobNum,MachineNum,JobOpNum,ProcessingTime):
    file = open('data.txt','w')
    file.write(f"{JobNum} {MachineNum} ")
    for i in range(JobNum):
        file.write(f"{JobOpNum[i]} ")
    for i in range(JobNum):
        for j in range(JobOpNum[i]):
            for k in range(MachineNum):
                file.write(f"{ProcessingTime[i][j][k]} ")

def random_env():
    JobNum = random.randint(0, 4)
    MachineNum = random.randint(0, JobNum)
    JobNum = JobNums[JobNum]
    MachineNum = MachineNums[MachineNum]
    JobOpNum = [random.randint(3, 8) for i in range(JobNum)]
    ProcessingTime = np.zeros([JobNum, 8, MachineNum])
    for i in range(JobNum):
        for j in range(JobOpNum[i]):
            for k in range(MachineNum):
                ProcessingTime[i][j][k] = random.randint(2, 10)
    ret = Situation(JobNum, MachineNum, JobOpNum, ProcessingTime)
    # writefile(JobNum,MachineNum,JobOpNum,ProcessingTime)
    return ret


def static_env():
    data = np.loadtxt('data.txt',dtype=int,delimiter=' ')
    JobNum = data[0]
    MachineNum = data[1]
    JobOpNum = np.zeros([JobNum],dtype=int)
    ProcessingTime = np.zeros([JobNum, 8, MachineNum])
    for i in range(JobNum):
        JobOpNum[i] = data[i + 2]
    now = JobNum + 2
    for i in range(JobNum):
        for j in range(JobOpNum[i]):
            for k in range(MachineNum):
                ProcessingTime[i][j][k] = data[now]
                now += 1
    ret = Situation(JobNum, MachineNum, JobOpNum, ProcessingTime)
    return ret


def create_env(args):
    if not args.data:
        ret = random_env()
    else:
        ret = static_env()
    print("env info:")
    print(f"JobNum:{ret.JobNum}")
    print(f"MachNum:{ret.MachNum}")
    print("JobOpNum:")
    print(ret.JobOpNum)
    return ret


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    args = get_common_args()
    args = get_mixer_args(args)
    env = create_env(args)
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    runner = Runner(env, args)
    if not args.evaluate:
        runner.run(1)
    else:
        average_spend_time, _ = runner.evaluate()
        print('The average spend time of model is  {}'.format(average_spend_time))
