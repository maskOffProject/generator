import Daugmentation
from keras.utils.data_utils import Sequence
import numpy as np


class MaSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, Daugmentation):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = Daugmentation

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        ts = self.augment.get_ts(len(batch_x))
        return np.array([self.augment.aug(x, t) for x, t in zip(batch_x, ts)]), np.array(
            [self.augment.aug(y, t) for y, t in zip(batch_y, ts)])