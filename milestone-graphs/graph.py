import matplotlib.pyplot as plt
import numpy as np

def parse_lines(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    # print(lines[0].split("\t"))

    # ['Epoch', 'AverageEpRet', 'StdEpRet', 'MaxEpRet', 'MinEpRet', 'EpLen', 
    # 'AverageVValStdVVals', 'MaxVVals', 'MinVVals', 'TotalEnvInteracts', 'LossPi', 
    # 'LossV', 'DeltaLossPiDeltaLossV', 'Entropy', 'Risk', 'ExpectedRet', 'KL', 'Time']

    epoch = [] # 0
    avg_ep_ret = [] # 1

    # risk = [] # -4
    exp_ret = [] # -3

    for line in lines[1:]:
        line_split = [float(x.strip()) for x in line.split("\t")]
        # print(line_split)

        epoch.append(int(line_split[0]))
        avg_ep_ret.append(line_split[1])
        # exp_ret.append(line_split[-3])

    # print(epoch)
    # print(avg_ep_ret)
    # print(np.average(risk))
    # print(np.average(exp_ret))

    return epoch, avg_ep_ret 
    # return epoch, exp_ret
    # return risk, exp_ret 

################################### GRAPH CARTPOLE ###################################
def graph_cartpole():

    epoch0, av_ep_ret0 = parse_lines("data/cartpole_lambda0/cartpole_lambda0_s0/progress.txt")
    plt.plot(epoch0, av_ep_ret0, color='b', label = 'lambda=0')

    epoch75, av_ep_ret75 = parse_lines("data/cartpole_lambda0,75/cartpole_lambda0,75_s0/progress.txt")
    plt.plot(epoch75, av_ep_ret75, color='r', label = 'lambda=0.75')

    epoch90, av_ep_ret90 = parse_lines("data/cartpole_lambda0,9/cartpole_lambda0,9_s0/progress.txt")
    plt.plot(epoch90, av_ep_ret90, color='g', label = 'lambda=0.9')

    epoch98, av_ep_ret98 = parse_lines("data/cartpole_lambda0,98/cartpole_lambda0,98_s0/progress.txt")
    plt.plot(epoch98, av_ep_ret98, color='m', label = 'lambda=0.98')

    epoch1, av_ep_ret1 = parse_lines("data/cartpole_lambda1/cartpole_lambda1_s0/progress.txt")
    plt.plot(epoch1, av_ep_ret1, color='c', label = 'lambda=1')

    plt.xlabel('Epoch')
    plt.ylabel('Average Episode Return')
    plt.title('Cartpole Task: Average Episode Return vs Epoch')
    plt.legend(loc='center right', prop={'size': 7})
    plt.savefig('cartpole.png')

################################### GRAPH EXP VS RISK ###################################
def graph_cartpole3():

    epoch0, exp_ret0 = parse_lines("data/cartpole_lambda0/cartpole_lambda0_s0/progress.txt")
    plt.plot(epoch0, exp_ret0, color='b', label = 'lambda=0')

    epoch75, exp_ret75 = parse_lines("data/cartpole_lambda0,75/cartpole_lambda0,75_s0/progress.txt")
    plt.plot(epoch75, exp_ret75, color='r', label = 'lambda=0.75')

    epoch90, exp_ret90 = parse_lines("data/cartpole_lambda0,9/cartpole_lambda0,9_s0/progress.txt")
    plt.plot(epoch90, exp_ret90, color='g', label = 'lambda=0.9')

    epoch98, exp_ret98 = parse_lines("data/cartpole_lambda0,98/cartpole_lambda0,98_s0/progress.txt")
    plt.plot(epoch98, exp_ret98, color='m', label = 'lambda=0.98')

    epoch1, exp_ret1 = parse_lines("data/cartpole_lambda1/cartpole_lambda1_s0/progress.txt")
    plt.plot(epoch1, exp_ret1, color='c', label = 'lambda=1')

    plt.xlabel('Epoch')
    plt.ylabel('Expected Return')
    plt.title('Cartpole Task: Expected Return vs Epoch')
    plt.legend(loc='center right', prop={'size': 7})
    plt.savefig('cartpole3.png')
#######################################################################################
graph_cartpole()
# graph_cartpole3()