import random

from torch.utils.data.sampler import Sampler

class UniformSampler(Sampler):
    def __init__(self, label_indices):
        self.label_indices = label_indices
        self.num_classes = len(label_indices)
        self.num_samples_per_class = min([len(indices) for indices in self.label_indices.values()])
        self._num_samples = self.num_classes * self.num_samples_per_class

    @property
    def num_samples(self):
        return self._num_samples

    def __iter__(self):
        uniform_indices = []
        for label in self.label_indices:
            uniform_indices.extend(random.sample(self.label_indices[label], self.num_samples_per_class))

        return iter(uniform_indices)

    def __len__(self):
        return self.num_samples
