import numpy as np
import matplotlib.pyplot as plt

# formula
x = np.linspace(-5, 5, 300)
y = np.sin(3*x)


plt.figure(1)
plt.plot(x,y)
plt.title("sin wave(3/period)")
plt.xlabel('x')
plt.ylabel('y')
# plt.xlim(-10,10)
# plt.ylim(-0.1,1.2)
plt.show()

# output sinwave
path_psf = r"/home/bt/文件/bosi_optics/DPM_verify/sinwave.txt"
with open(path_psf, "wt") as f:
    for i in range(len(x)):
        f.write(str(x[i])+"\t"+str(y[i])+"\n")