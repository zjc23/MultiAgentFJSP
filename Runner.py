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
        self.average_spend_times = []
        self.episode_rewards = []

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        while time_steps < self.args.n_steps:
            print('Run {}, time_steps {}'.format(num, time_steps))
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                average_spend_time, episode_reward = self.evaluate()
                print(f'average_spend_time is {average_spend_time}')
                print(f'episode_reward is {episode_reward}')
                self.average_spend_times.append(average_spend_time)
                self.episode_rewards.append(episode_reward)
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
        average_spend_time, episode_reward = self.evaluate()
        print('average_spend_time is ', average_spend_time)
        self.average_spend_times.append(average_spend_time)
        self.episode_rewards.append(episode_reward)
        self.plt(num)

    def evaluate(self):
        spend_time = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, time = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            spend_time += time
        return spend_time / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        plt.figure()
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.average_spend_times)), self.average_spend_times)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('average_spend_time')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.tight_layout()
        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.average_spend_times)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        plt.close()









