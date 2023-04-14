from torch.utils.data import Dataset
import random
from collections import deque
import numpy as np
import torch

class ReplayMemory(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.position = 0

    def push(self, item):
        # replace old experience with new experience
        if len(self.memory) < self.position + 1:
            self.memory.append(item)
        else:
            self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    # def sample(self, batch_size):
    #     # total = len(self.memory)
    #     # indices = np.random.choice(total, batch_size, replace=False)
    #     # samples = [self.memory[idx] for idx in indices]
    #     samples = self.memory
    #     return samples


    def is_full(self):
        return len(self.memory) == self.capacity

    def __getitem__(self, item):
        return self.memory[item]

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = list()


# prioritized RB
class PrioritizedReplayMemory(Dataset):
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.memory = list()
        # self.memory =deque(maxlen=capacity)
        # self.priorities = deque(maxlen=capacity)
        self.priorities = list()
        self.position = 0
    
    def push(self, item):
        max_prio = max(self.priorities) if self.priorities else 1.0
        priority = max_prio if not self.priorities else min(self.priorities)
        if len(self.memory) == self.capacity:
            self.memory[self.position] = item
            self.priorities[self.position] = priority
        else:
            self.memory.append(item)
            self.priorities.append(priority)
        self.position = (self.position + 1) % self.capacity
    
    # def sample(self, batch_size):
    #     total = len(self.memory)
    #     probs = [p ** self.alpha for p in self.priorities]
    #     probs = np.array(probs) / sum(probs)
    #     indices = np.random.choice(total, batch_size, p=probs)
    #     # probs = torch.tensor(probs) / torch.sum(torch.tensor(probs))
    #     # indices = torch.multinomial(probs, batch_size, replacement=True)
    #     '''
    #     np.random.choice(total, batch_size, p=probs) 表示从长度为 total 的序列中，随机选取 batch_size 个元素，概率分布为 probs，返回选取的元素的下标。
    #     其中 probs 是概率列表，需要满足概率之和为 1。选取的过程是带权重的，即概率越大的元素被选中的概率越大。
    #     '''
    #     samples = [self.memory[idx] for idx in indices]

    #     return samples
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha

    # def split_by_priorities(self, batch_size):
    #     priorities = np.array(self.priorities[:self.position])
    #     probs = priorities ** self.alpha
    #     probs /= probs.sum()
    #     num_batches = int(np.ceil(len(self.memory) / batch_size))
    #     batch_sizes = [batch_size] * (num_batches - 1) + [len(self.memory) % batch_size]

    #     indices = np.arange(len(self.memory))
    #     subsets = []
    #     for batch_size in batch_sizes:
    #         batch_indices = np.random.choice(indices, size=batch_size, p=probs, replace=True)
    #         subset = PrioritizedReplayMemory(capacity=len(batch_indices), alpha=self.alpha)
    #         for idx in batch_indices:
    #             subset.push(self.memory[idx])
    #             subset.update_priorities([subset.position - 1], [self.priorities[idx]])
    #         subsets.append(subset)

    #     return subsets
    
    def is_full(self):
        return len(self.memory) == self.capacity

    def __getitem__(self, item):
        return self.memory[item]

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = list()
        self.priorities.clear()


from torch.utils.data import DataLoader
import random
import math
class PrioritizedReplayBufferDataLoader(DataLoader):
    def __init__(self, memory, batch_size, alpha=0.6):
        self.memory = memory
        self.batch_size = batch_size
        self.alpha = alpha

        super().__init__(memory, batch_size=batch_size, shuffle=True)

    def __iter__(self):
        probs = [(p + 1e-5) ** self.alpha for p in self.memory.priorities]
        probs = torch.tensor(probs) / torch.sum(torch.tensor(probs))

        indices = torch.multinomial(probs, len(self.memory), replacement=True)
        weights = (len(self.memory) * probs[indices]).pow(-self.alpha)
        weights = weights / torch.max(weights)

        batch = [self.memory[idx] for idx in indices.tolist()]

        yield batch
# class PrioritizedReplayBufferDataLoader(DataLoader):
#     def __init__(self, buffer, batch_size, alpha=0.6):
#         self.buffer = buffer
#         self.batch_size = batch_size
#         self.alpha = alpha

#        # 划分数据集
#         # total_size = len(buffer)
#         self.subsets = buffer.split_by_priorities(batch_size)

#         super().__init__(buffer, batch_size=batch_size, shuffle=False)

#     def __iter__(self):
#         for subset in self.subsets:
#             # 计算样本权重
#             probs = [(p + 1e-5) ** self.alpha for p in subset.priorities]
#             probs = torch.tensor(probs) / torch.sum(torch.tensor(probs))

#             # 选择样本并计算权重
#             indices = torch.multinomial(probs, self.batch_size, replacement=True)
#             weights = (len(subset) * probs[indices]).pow(-self.alpha)
#             weights = weights / torch.max(weights)

#             # 根据索引获取批次数据
#             batch = [subset[idx] for idx in indices.tolist()]

#             yield batch