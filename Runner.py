import numpy as np
import os
from Simulator import RolloutWorker
from Agent import Agents
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, env, args):
        self.env = env
        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        """
        if not args.evaluate and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)
        """
        self.args = args
        self.spend_times = []
        self.episode_rewards = []
        self.total_spend_times = 0
        self.total_rewards = 0
        self.average_spend_times = []
        self.average_rewards = []

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        while time_steps < self.args.n_steps:
            print('Run {}, time_steps {}'.format(num, time_steps))
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                spend_time, episode_reward = self.evaluate()
                print(f'spend_time is {spend_time}')
                print(f'episode_reward is {episode_reward}')
                self.spend_times.append(spend_time)
                self.total_spend_times += spend_time
                self.average_spend_times.append(self.total_spend_times / len(self.spend_times))
                self.episode_rewards.append(episode_reward)
                self.total_rewards += episode_reward
                self.average_rewards.append(self.total_rewards / len(self.episode_rewards))
                self.plt(num)
                evaluate_steps += 1
            episodes = []
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, _, steps = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                time_steps += steps
                # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
            train_steps += 1
        spend_time, episode_reward = self.evaluate()
        print(f'spend_time is {spend_time}')
        self.spend_times.append(spend_time)
        self.total_spend_times += spend_time
        self.average_spend_times.append(self.total_spend_times / len(self.spend_times))
        self.episode_rewards.append(episode_reward)
        self.total_rewards += episode_reward
        self.average_rewards.append(self.total_rewards / len(self.episode_rewards))
        self.plt(num)

    def evaluate(self):
        _, episode_reward, time = self.rolloutWorker.generate_episode(0, evaluate=True)
        return time, episode_reward

    def plt(self, num):
        plt.figure()
        plt.plot(range(len(self.spend_times)), self.spend_times)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('spend_time')
        plt.savefig(self.save_path + '/plt_time.png', format='png')

        plt.clf()
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')
        plt.savefig(self.save_path + '/plt_reward.png', format='png')

        plt.clf()
        plt.plot(range(len(self.average_rewards)), self.average_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('average_rewards')
        plt.savefig(self.save_path + '/plt_average_reward.png', format='png')

        plt.clf()
        plt.plot(range(len(self.average_spend_times)), self.average_spend_times)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('average_times')
        plt.savefig(self.save_path + '/plt_average_time.png', format='png')
        print("?")

        # np.save(self.save_path + '/win_rates_{}'.format(num), self.average_spend_times)
        # np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        plt.close()









