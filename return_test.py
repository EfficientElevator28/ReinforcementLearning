
import matplotlib.pyplot as plt
import numpy as np

# ===============================================================
def moving_average(data, window_size=100): #used this approach https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
    """
    Calculates a moving average for all the data
    Args:
        data: set of values
        window_size: number of data points to consider in window
    Returns:
        Moving average of the data
    """    
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec
# ===============================================================

GAMMA = .95
reward = -10
returns = 0

return_at_step = []
reward_at_step = []

for i in range(10000):
    reward_at_step.append(reward)
    returns = reward + returns*GAMMA
    return_at_step.append(returns)


plt.figure()
plt.plot(return_at_step)
plt.plot(moving_average(return_at_step))
plt.xlabel("Steps")
plt.ylabel("Return")

plt.figure()
plt.plot(reward_at_step)
plt.plot(moving_average(reward_at_step))
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.show()