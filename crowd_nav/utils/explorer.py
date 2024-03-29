import os
import logging
import copy
import torch
from tqdm import tqdm
from crowd_sim.envs.utils.info import *
from crowd_nav.policy.reward_estimate import Reward_Estimator
from crowd_sim.envs.utils.state import tensor_to_joint_state
from collections import defaultdict
class Explorer(object):
    def __init__(self, env, robot, device, writer, use_noisy_net, memory=None, gamma=None, target_policy=None,
                    intrinsic_reward = None):
        self.env = env
        self.robot = robot
        self.device = device
        self.writer = writer
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.statistics = None
        if use_noisy_net is not None:
            self.use_noisy_net = use_noisy_net
        self.intrinsic_reward_alg = intrinsic_reward

    def set_use_noisy_net(self, use_noisy):
        self.use_noisy_net = use_noisy

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None, epoch=None,
                       print_failure=False,reward_estimator = None):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        discomfort = 0
        min_dist = []
        cumulative_rewards = []
        average_returns = []
        returns_list = []
        collision_cases = []
        timeout_cases = []
        discomfort_nums = []
        if phase in ['test', 'val'] or imitation_learning:
            pbar = tqdm(total=k)
        else:
            pbar = None
        if self.robot.policy.name in ['model_predictive_rl', 'tree_search_rl']:
            # train()和eval()是nn.Module内置的两个函数，用于设置网络的训练和测试模式
            if phase in ['test', 'val'] and self.use_noisy_net:
                self.robot.policy.model[2].eval()  # 设置模式，不进行forward
                # self.model = [graph_model1, graph_model2, self.value_estimator.value_network,
                #              self.state_predictor.human_motion_predictor]
                # value_network = DuelingDQN(config.gcn.X_dim, self.action_num)
                # DuelingDQN是一个继承自nn.Module的深度神经网络，用于实现Dueling DQN算法
            else:
                self.robot.policy.model[2].train() # batchnorm dropout应用

        for i in range(k):
            ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []
            dones = []
            num_discoms =[]
            robot_states_seen = defaultdict(int)
            while not done:
                num_discom = 0
                action, action_index = self.robot.act(ob) # self.policy.predict 输入ob（人类状态）和机器人的所有属性，包括目标
                ob, reward, done, info = self.env.step(action)# 执行action动作并更新环境的状态
                states.append(self.robot.policy.last_state) # predict(state)机器人的所有属性(包括目标),ob（人类状态）
                # if(self.robot.policy.name in ['TD3RL']):
                #     actions.append(torch.tensor((action.vx, action.vy)).float())
                # else:
                full_state = self.robot.get_state(ob)
                robot_state = (full_state[0][0][0],full_state[0][0][1],full_state[1][:][0],full_state[1][:][0])
                # print("full_state",full_state)
                if robot_state in robot_states_seen:
                    robot_states_seen[robot_state] += 1
                else:
                    robot_states_seen[robot_state] = 1

                # for TD3rl, append the velocity and theta
                actions.append(action_index)
                # rewards.append(reward)
                # actually, final states of timeout cases is not terminal states
                # 因为我要改成her的，所以把done的状态true
                if isinstance(info, Timeout):
                    dones.append(False)
                else:
                    dones.append(done)
                rewards.append(reward)
                if isinstance(info, Discomfort):
                    discomfort += 1
                    min_dist.append(info.min_dist)
                    num_discom = info.num
                num_discoms.append(num_discom)
            # add the terminal state 多增加的一个robot状态和ob
            states.append(self.robot.get_state(ob))  # state = JointState(self.get_full_state(), ob)

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
                if phase in ['test']:
                    print('collision happen %f', self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                if phase in ['test']:
                    print('timeout happen %f', self.env.global_time)
                    rewards[-1] = rewards[-1]-0.25
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                # if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set   
                if self.intrinsic_reward_alg is not None:
                    embeddings = None
                    if self.intrinsic_reward_alg.name == "RE3":
                        embeddings = []
                        intrinsic_rewards = rewards
                        for i_intrinsic in range(len(rewards)):
                            if type(states[i_intrinsic]) is tuple:
                                embeddings.append(self.intrinsic_reward_alg.get_embeddings((states[i_intrinsic][0].unsqueeze(0), states[i_intrinsic][1].unsqueeze(0))).to(states[0][0].device))
                            else:
                                embeddings.append(self.intrinsic_reward_alg.get_embeddings(states[i_intrinsic].unsqueeze(0).to(states[0].device)))
                    else:
                        intrinsic_rewards = []
                        for i_intrinsic in range(len(rewards)):
                            intrinsic_rewards.append(rewards[i_intrinsic] + self.intrinsic_reward_alg.compute_intrinsic_reward(
                                tuple(torch.unsqueeze(substate, 0) for substate in states[i_intrinsic]),
                                tuple(torch.unsqueeze(substate, 0) for substate in states[i_intrinsic + 1]), 
                                actions[i_intrinsic],robot_states_seen))                         
                    self.update_her_memory(states, actions, intrinsic_rewards, dones,info,reward_estimator,robot_states_seen, imitation_learning, embeddings)
                else:
                    self.update_her_memory(states, actions, rewards, dones,info,reward_estimator, imitation_learning)
            discomfort_nums.append(sum(num_discoms))
            # cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
            #                                * reward for t, reward in enumerate(rewards)]))
            cumulative_rewards.append(sum(rewards))
            returns = []
            for step in range(len(rewards)):
                step_return = sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                   * reward for t, reward in enumerate(rewards[step:])])
                returns.append(step_return)
            returns_list = returns_list + returns
            average_returns.append(average(returns))

            if pbar:
                pbar.update(1)




        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        extra_info = extra_info + '' if epoch is None else extra_info + ' in epoch {} '.format(epoch)
        logging.info('{:<5} {}has success rate: {:.3f}, collision rate: {:.3f}, nav time: {:.3f}, total reward: {:.4f},'
                     ' average return: {:.4f}'. format(phase.upper(), extra_info, success_rate, collision_rate,
                                                       avg_nav_time, sum(cumulative_rewards),
                                                       average(average_returns)))
        # if phase in ['val', 'test'] or imitation_learning:
        total_time = sum(success_times + collision_times + timeout_times) / self.robot.time_step
        logging.info('Frequency of being in danger: %.3f and average min separate distance in danger: %.2f',
                    discomfort / total_time, average(min_dist))
        logging.info('discomfor nums is %.0f and return is %.04f and length is %.0f', sum(discomfort_nums),
                     average(returns_list), len(returns_list))
        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

        self.statistics = success_rate, collision_rate, avg_nav_time, sum(cumulative_rewards), average(average_returns), discomfort, total_time,robot_states_seen

        return self.statistics

    def update_memory(self, states, actions, rewards, dones, imitation_learning=False, embeddings = None):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')
        
        for i, state in enumerate(states[:-1]):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                action = actions[i]
                done = torch.Tensor([dones[i]]).to(self.device)
                next_state = self.target_policy.transform(states[i+1])
                value = sum([pow(self.gamma, (t - i) * self.robot.time_step * self.robot.v_pref) * reward *
                             (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                next_state = states[i+1]
                action = actions[i]
                # if i == len(states) - 1:
                if i == len(states) - 2:
                    # terminal state
                    value = reward
                else:
                    value = 0
                value = torch.Tensor([value]).to(self.device)
                reward = torch.Tensor([rewards[i]]).to(self.device)
                done = torch.Tensor([dones[i]]).to(self.device)
            if self.intrinsic_reward_alg is not None and self.intrinsic_reward_alg.name == "RE3":
                if self.target_policy.name == 'ModelPredictiveRL' or self.target_policy.name == 'TreeSearchRL' or self.target_policy.name == 'TD3RL':
                    self.memory.push((state[0], state[1], action, value, done, reward, next_state[0], next_state[1], embeddings[i]))
                else:
                    self.memory.push((state, value, done, reward, next_state, embeddings[i]))
            elif self.target_policy.name == 'ModelPredictiveRL' or self.target_policy.name == 'TreeSearchRL' or self.target_policy.name == 'TD3RL':
                self.memory.push((state[0], state[1], action, value, done, reward, next_state[0], next_state[1]))
            else:
                self.memory.push((state, value, done, reward, next_state))


    def update_her_memory(self, states, actions, rewards, dones,info,reward_estimator,robot_states_seen, imitation_learning=False,embeddings = None):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')
        for i, state in enumerate(states[:-1]):
            reward = rewards[i]
            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                action = actions[i]
                done = torch.Tensor([dones[i]]).to(self.device)
                next_state = self.target_policy.transform(states[i+1])
                value = sum([pow(self.gamma, (t - i) * self.robot.time_step * self.robot.v_pref) * reward *
                             (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:          
                next_state = states[i+1]
                action = actions[i]
                if i == len(states) - 2: # 源代码错误的，永远不可能到len -1
                    # terminal state
                    value = reward
                else:
                    value = 0
                
            if isinstance(info, Timeout):
                # 加入新的目标
                new_goal = states[-1]
                # if i ==0 :
                #     print("state1",state)
                #     print("new_goal",new_goal)
                # print("state_len",len(states))
                # print("new_goal",new_goal[0][0])
                state[0][0][5] = new_goal[0][0][0]  # gx = px
                state[0][0][6] = new_goal[0][0][1]  # gy = py
                next_state[0][0][5] = new_goal[0][0][0]  # gx
                next_state[0][0][6] = new_goal[0][0][1]  # gy
                # if i ==0 :
                #     print("state2",state)   
                if i == len(states) - 2 :
                    # print("state_len",len(states))
                    print("timeout",i)
                    # 成功的奖励
                    reward = 0
                    value = torch.Tensor([reward]).to(self.device)
                    reward = torch.Tensor([reward]).to(self.device)
                    done = torch.Tensor([True]).to(self.device)                  
                else:
                    # print("to_time",i)
                    update_state = tensor_to_joint_state((state[0], state[1]))
                    update_next_state = tensor_to_joint_state((next_state[0], next_state[1]))
                    reward , _ = reward_estimator.estimate_reward_on_predictor(update_state,update_next_state) # 输出reward和info
                    # print("reward",reward)
                    reward = reward + self.intrinsic_reward_alg.compute_intrinsic_reward(
                        tuple(torch.unsqueeze(substate, 0) for substate in state),
                        tuple(torch.unsqueeze(substate, 0) for substate in next_state),
                                actions[i],robot_states_seen)
                    value = 0
                    value = torch.Tensor([value]).to(self.device)
                    reward = torch.Tensor([reward]).to(self.device)
                    done = torch.Tensor([dones[i]]).to(self.device)
            else:
                value = torch.Tensor([value]).to(self.device)
                reward = torch.Tensor([rewards[i]]).to(self.device)
                done = torch.Tensor([dones[i]]).to(self.device)

            if self.intrinsic_reward_alg is not None and self.intrinsic_reward_alg.name == "RE3":
                if self.target_policy.name == 'ModelPredictiveRL' or self.target_policy.name == 'TreeSearchRL' or self.target_policy.name == 'TD3RL':
                    self.memory.push((state[0], state[1], action, value, done, reward, next_state[0], next_state[1], embeddings[i]))
                else:
                    self.memory.push((state, value, done, reward, next_state, embeddings[i]))
            elif self.target_policy.name == 'ModelPredictiveRL' or self.target_policy.name == 'TreeSearchRL' or self.target_policy.name == 'TD3RL':              
                self.memory.push((state[0], state[1], action, value, done, reward, next_state[0], next_state[1]))
            else:                  
                self.memory.push((state, value, done, reward, next_state))




    def log(self, tag_prefix, global_step):
        sr, cr, time, reward, avg_return,_,_ ,_= self.statistics
        self.writer.add_scalar(tag_prefix + '/success_rate', sr, global_step)
        self.writer.add_scalar(tag_prefix + '/collision_rate', cr, global_step)
        self.writer.add_scalar(tag_prefix + '/time', time, global_step)
        self.writer.add_scalar(tag_prefix + '/reward', reward, global_step)
        self.writer.add_scalar(tag_prefix + '/avg_return', avg_return, global_step)


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
