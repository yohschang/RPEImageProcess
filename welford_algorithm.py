import numpy as np

class OnlineVariance(object):
    """
    Welford's algorithm computes the sample variance incrementally.
    """

    def __init__(self, iterable=None, ddof=1):
        self.ddof, self.n, self.mean, self.M2 = ddof, 0, 0.0, 0.0
        if iterable is not None:
            for datum in iterable:
                self.include(datum)

    def include(self, datum):
        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
        self.M2 += self.delta * (datum - self.mean)

    @property
    def variance(self):
        return self.M2 / (self.n - self.ddof)

    @property
    def std(self):
        return np.sqrt(self.variance)


# N = 100
# data = np.random.random(N)
# ov = OnlineVariance(ddof=0)
# for d in data:
#     ov.include(d)
# std = ov.std
# print(std)
#
# if (std == data.std()):
#     print("True")
# # assert np.allclose(std, data.std())


