MAFS5370 Assignment 1
Group 3 (LAI Fujie and CHENG Yu)

All scripts are saved in script folder. The script/hw1_ddpg_jupyter_notebook_sample.ipynb contains samples of how to use
the class.

Below listed out some key package version used in the notebook sample:
- Python version: 3.9
- gym: 0.21.0

Below are introductions of our approach

DDPG On assetallocation

The reason for choosing ddpg:
We find that although there are only 10-time steps in the asset allocation problem, the actions space(choose the weight for the risky asset at each step)
and so the states space(wealth at each step) are not discrete. We can choose continues actions from 0 to 1,which will lead to any states at next step.
DDPG (Deep Deterministic Policy Gradient) is a reinforcement learning algorithm that is particularly well-suited for continuous action spaces, where actions
can take on a large range of values. DDPG is a model-free, off-policy algorithm that combines ideas from both Q-learning and policy gradient methods.


Environment Design Using Openai Jym:
We choose to use openai's jym package(version 0.21.0) to do reinforcement learnign training. In the file "Assetallocate.py" we design our custom environment.
Function introductions are below:
_init_: define inital wealth, probability of up and down, return,time step
step: define reward and the next step state when it receives an action
reset: required by jym package
render:required by jym package


Deep Neural Network  Structure:
There are four neural network in DDPG. Two are Policynet, also called "actor",and the ohther are Qvaluenet, also called "critic".The actor network takes in the current state
of the environment and outputs a continuous action that the agent should take. The critic network takes in the current state and action, and outputs an estimate of the
 expected future reward (i.e., the Q-value) for that state-action pair.DDPG also incorporates target networks to help stabilize the learning process. The target networks are
essentially copies of the actor and critic networks that are periodically updated to slowly track the learned networks. This helps prevent the target values from oscillating or
diverging during training, which can happen when using only a single network.

Policynet is a one hidden layer network,input is state and output is action.We use sigmoid function to generate a 0 to 1 action.
Qvaluenet is a two hidden layers network, input is action and ouput is expected return/reward.


Action、Reward、State and Q value  in DDPG:
Action: Choose continues actions from 0 to 1. In DDPG, we use Policynet to generate action, and also we add some gaussian noise to it for exploring more in the environment,
Reward : 0 if t from 0 to 9 and expexted reward given by critic network if t == 10. We use utility function (1-exp(-a*wealth))/a as reward, a is a constant (we set 1).
State: Wealth, we initialize it as 1 at t==0, and wt = xt * wt * risky_return + (1 - xt) * wt * self.riskfree_return;risky_return has probability p to go up A and 1-p to go down B.
Q value: The finaly Q value we represent is generated by trained-well target_Qvaluenet,  in the code, Qvalue== reward + gamma*(next_state_Qvalue)*(1 - dones).
