# 机器 工件
class Object:
    def __init__(self, ID):
        self.ID = ID
        self.Start = []
        self.End = []
        self.AssignFor = []

    def _add(self, S, E, obs):
        self.Start.append(S)
        self.End.append(E)
        self.AssignFor.append(obs)

    def idle_time(self):
        Idle = []
        try:
            if self.Start[0] != 0:
                Idle.append([0, self.Start[0]])
            K = [[self.End[i], self.Start[i + 1]] for i in range(len(self.End)) if self.Start[i + 1] - self.End[i] > 0]
            Idle.extend(K)
        except:
            pass
        return Idle

    def is_idle(self,t):
        if len(self.Start) <= 0:
            return 1
        if self.Start[-1] >= t or t >= self.End[-1]:
            return 1
        return 0

    def left_time(self,t):
        if len(self.End) <= 0:
            return 0
        if self.End[-1] - t > 0:
            return self.End[-1] - t
        return 0

    def reset(self):
        self.Start = []
        self.End = []
        self.AssignFor = []

    def latest_time(self):
        if len(self.Start) <= 0:
            return 0
        return self.End[-1]