import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    @torch.no_grad()
    def generate_episode(self, episode_num=None, evaluate=False):
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            state = self.env.get_global_state()
            obs = self.env.get_observation()
            actions = np.full(self.n_agents, -1)
            avail_actions, actions_onehot = [], []
            attempt_to_explore = np.zeros(self.args.n_agents)
            all = []
            calcu = self.env.get_idle_agents()
            # 每个智能体获取信息
            for agent_id in range(self.n_agents):
                if calcu[agent_id] < 0:
                    actions[agent_id] = int(self.args.n_actions + calcu[agent_id])
                    avail_action = np.zeros(self.args.n_actions)
                    avail_action[actions[agent_id]] = 1
                    avail_actions.append(avail_action)
                    continue
                avail_action = self.env.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_action)
                # avail_action[self.args.n_actions - 3] = 0
                q_value = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon)
                # print(f"agent{agent_id}:{q_value}")
                if np.random.uniform() < epsilon:
                    attempt_to_explore[agent_id] = 1
                    continue
                # generate onehot vector of th action
                for j in range(self.args.n_actions):
                    ins = (q_value[0][j], agent_id, j)
                    all.append(ins)
            # 根据q值调度
            all = sorted(all, key=lambda x: x[0], reverse=True)
            for i, item in enumerate(all):
                qval, agent, act = item[0], item[1], item[2]
                # print(f"qval:{qval} act:{act} agent:{agent}")
                if actions[agent] != -1:
                    continue
                if act == self.args.n_actions - 3 or (actions[agent] == -1 and (act not in actions)):
                    actions[agent] = act
            # 处理探索
            for i in range(self.args.n_agents):
                if attempt_to_explore[i] == 1:
                    avail_action = self.env.get_avail_agent_actions(i)
                    for j in range(self.args.n_actions - 3):
                        if j in actions:
                            avail_action[j] = 0
                    avail_action_ind = np.nonzero(avail_action)[0]
                    act = np.random.choice(avail_action_ind)
                    actions[i] = act
            # 记录数据
            for i in range(self.args.n_agents):
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[actions[i]] = 1
                actions_onehot.append(action_onehot)
                # print(action_onehot)
            # print(actions_onehot)
            # print(actions)
            reward, terminated = self.env.scheduling(actions)
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step = self.env.Time
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # print(f"该轮奖励和为{episode_reward}")
        # last obs
        obs = self.env.get_observation()
        state = self.env.get_global_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        return episode, episode_reward, step
