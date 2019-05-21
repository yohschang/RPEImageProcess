import numpy as np
from PSF_calculate import read_profile, normalize, runnung_mean
import matplotlib.pyplot as plt

x, y = read_profile(r'E:\DPM\20190508\proflie.txt')

plt.figure()
plt.plot(x, y)
plt.show()

dx = 5
dy = np.diff(y)/dx

plt.figure()
plt.plot(x[:719], dy)
plt.xlim(320,340)
plt.show()
