from fjspObject import Object
import numpy as np
import torch
from copy import deepcopy

class Situation:
    def __init__(self,JobNum,MachNum,JobOpNum,ProcessingTime):
        self.Terminated = False
        self.JobNum = JobNum
        self.MachNum = MachNum
        self.JobOpNum = JobOpNum
        self.ProcessingTime = ProcessingTime # 工件i的第j道操作在第k个机器所需时间
        self.Jobs = []
        for i in range(JobNum):
            J = Object(i)
            self.Jobs.append(J)
        self.Machines = []
        for i in range(MachNum):
            M = Object(i)
            self.Machines.append(M)
        self.Time = 0
        self.n_agents = JobNum
        self.n_actions = MachNum + 3
        self.C = 100
        self.Cup = 150
        self.Usage = 1

    def scheduling(self,actions):
        # print(f"第{self.Time}时刻：采取动作{actions}")
        for i, item in enumerate(actions):
            if actions[i] >= self.MachNum:
                continue
            # print(f"为{i}号工件分配机器{actions[i]}")
            Machine = self.Machines[actions[i]]
            Workpiece = self.Jobs[i]
            Machine._add(self.Time,self.Time + self.ProcessingTime[Workpiece.ID][len(Workpiece.Start)][Machine.ID],Workpiece.ID)
            Workpiece._add(self.Time,self.Time + self.ProcessingTime[Workpiece.ID][len(Workpiece.Start)][Machine.ID],Machine.ID)
        self.next_time()
        r = self.reward()
        return r, self.Terminated

    def get_observation(self):
        State = np.zeros([self.JobNum, self.MachNum])
        for i in range(self.JobNum):
            for j in range(self.MachNum):
                if len(self.Jobs[i].Start) >= self.JobOpNum[i]:
                        continue
                State[i][j] = self.ProcessingTime[i][len(self.Jobs[i].Start)][j]
        return State

    def get_global_state(self):
        State = np.zeros([(2 * self.JobNum + self.MachNum), self.MachNum])
        for i in range(self.JobNum):
            State[i][1] = self.Jobs[i].is_idle(self.Time)#是否在加工
            State[i][0] = self.Jobs[i].left_time(self.Time)#加工剩余时间
        for i in range(self.MachNum):
            State[self.JobNum + i][1] = self.Machines[i].is_idle(self.Time)#是否在加工
            State[self.JobNum + i][0] = self.Machines[i].left_time(self.Time)#加工剩余时间
        for i in range(self.JobNum):
            for j in range(self.MachNum):
                if len(self.Jobs[i].Start) < self.JobOpNum[i]:
                    State[self.JobNum + self.MachNum + i][j] = self.ProcessingTime[i][len(self.Jobs[i].Start)][j]#下一个操作在各个机器上的用时
        State = State.flatten()
        return State

    def next_time(self):
        self.Time = self.Time + 1
        for i in range(self.JobNum):
            if len(self.Jobs[i].Start) < self.JobOpNum[i]:
                # print(f"第{i}个工件未完工 {len(self.Jobs[i].Start)} {self.JobOpNum[i]}")
                return
            if self.Jobs[i].End[-1] > self.Time:
                # print(f"第{i}个工件未完工 {self.Jobs[i].Start[-1]} {self.Jobs[i].End[-1]}")
                return
        self.Terminated = True
        print(f"在t={self.Time}时完成该问题")

    def reward(self):
        Z = 100
        if self.Time > self.Cup:
            return -Z
        fin = self.guess_fin()
        r = self.C - fin
        self.C = fin
        return r

    def guess_fin(self):
        machines = deepcopy(self.Machines)
        jobs = deepcopy(self.Jobs)
        ti = self.Time
        while True:
            fin = True
            for i in range(self.JobNum):
                if len(jobs[i].Start) < self.JobOpNum[i]:
                    fin = False
            if fin:
                break
            for i in range(self.JobNum):
                if len(jobs[i].Start) >= self.JobOpNum[i] or (jobs[i].is_idle(ti) == 0):
                    continue
                machNum,valNum = -1, 0
                for j in range(self.MachNum):
                    if machines[j].is_idle(ti) == 0:
                        continue
                    if valNum < self.ProcessingTime[i][len(jobs[i].Start)][j]:
                        machNum, valNum = j, self.ProcessingTime[i][len(jobs[i].Start)][j]
                if machNum != -1:
                    jobs[i]._add(ti,ti + valNum,machNum)
                    machines[machNum]._add(ti,ti + valNum,i)
            ti += 1
        for i in range(self.JobNum):
            for j in range(len(jobs[i].Start)):
                # print(f"{jobs[i].Start[j]} {jobs[i].End[j]} {jobs[i].AssignFor[j]} / ",end = "")
                ti = max(ti,jobs[i].End[j])
            # print("")
        # print(f"{ti}")
        return ti


    def machine_usage(self):
        timesum = self.Time * self.MachNum
        timeusage = 0
        for i in self.Machines:
            for j in range(len(i.Start)):
                if i.End[j] <= self.Time:
                    timeusage += i.End[j] - i.Start[j]
                elif i.Start[j] <= self.Time:
                    timeusage += self.Time - i.Start[j]
        return timeusage / timesum


    def reset(self):
        self.Time = 0
        self.Terminated = False
        for i in self.Machines:
            i.reset()
        for i in self.Jobs:
            i.reset()
        self.C = 100

    def get_avail_agent_actions(self,ID):
        State = np.zeros(self.n_actions)
        if self.Jobs[ID].is_idle(self.Time) != 1:
            State[self.MachNum + 1] = 1
            return State
        elif len(self.Jobs[ID].Start) >= self.JobOpNum[ID]:
            State[self.MachNum + 2] = 1
            return State
        State[self.MachNum] = 1
        for i in range(self.MachNum):
            State[i] = self.Machines[i].is_idle(self.Time)
        return State

    def get_idle_agents(self):
        agents = np.zeros(self.JobNum)
        for i in range(self.JobNum):
            if self.Jobs[i].is_idle(self.Time) == 1 and len(self.Jobs[i].Start) < self.JobOpNum[i]:
                agents[i] = 1
            elif self.Jobs[i].is_idle(self.Time) == 0:
                agents[i] = -2
            elif len(self.Jobs[i].Start) >= self.JobOpNum[i]:
                agents[i] = -1
        return agents

    def get_env_info(self):
        n_actions = self.MachNum + 3# 不执行任何动作 因处理中而不做动作 因处理完不做动作
        n_agents = self.JobNum
        state_shape = (2 * self.JobNum + self.MachNum) * self.MachNum
        obs_shape = self.MachNum
        episode_limit = 1000
        info = dict(n_actions=n_actions,
                    n_agents=n_agents,
                    state_shape=state_shape,
                    obs_shape=obs_shape,
                    episode_limit=episode_limit)
        print(f"n_actions is {n_actions}")
        print(f"n_agents is {n_agents}")
        print(f"state_shape is {state_shape}")
        print(f"obs_shape is {obs_shape}")
        return info
