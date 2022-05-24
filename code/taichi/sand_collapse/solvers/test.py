import matplotlib.pyplot as plt
import numpy as np
fig = plt.gcf()
fig.show()
fig.canvas.draw()

for i in range(100):
    # compute something
    plt.imshow(np.random.rand(15,15)) # plot something
    
    # update canvas immediately
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.pause(0.01)
    fig.canvas.draw()