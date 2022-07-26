
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

net2 = pd.read_csv('run-tb_PPO_1-tag-eval_mean_reward.csv', usecols=['Step', 'Value'])
plt.plot(net2.Step, net2.Value, lw=1.5, label='PPO', color='pink')
net3 = pd.read_csv('run-tb_SAC_1-tag-eval_mean_reward.csv', usecols=['Step', 'Value'])
plt.plot(net3 .Step, net3 .Value, lw=1.5, label='SAC', color='green')


plt.legend(loc=0)
plt.xlabel('Total Steps')
plt.ylabel('Mean Evaluation Reward Value')

plt.grid()
plt.savefig('/home/ziqiao/RL/ERL_RL_Landing/experiments/learning/'+'compare_ppo_sac.png',dpi=600)
plt.show()