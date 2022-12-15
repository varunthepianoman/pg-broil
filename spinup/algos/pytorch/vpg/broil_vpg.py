import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.vpg.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup.rewards.cvar_utils import cvar_enumerate_pg
from spinup.rewards.cartpole_reward_utils import CartPoleReward
from spinup.rewards.pointbot_reward_utils import PointBotReward
# import dmc2gym
from custom_dmc2gym.dmc2gym.wrappers import DMCWrapper
import curl.utils

class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    #rew_dim is the dimensionality of the reward function posterior
    def __init__(self, obs_dim, act_dim, num_rew_fns, size, gamma=0.99, lam=0.95, image_size=100):
        self.num_rew_fns = num_rew_fns
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(core.combined_shape(size, num_rew_fns), dtype=np.float32)
        self.rew_buf = np.zeros(core.combined_shape(size, num_rew_fns), dtype=np.float32)
        self.ret_buf = np.zeros(core.combined_shape(size, num_rew_fns), dtype=np.float32)
        self.val_buf = np.zeros(core.combined_shape(size, num_rew_fns), dtype=np.float32)
        self.posterior_returns = []
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.image_size = image_size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=None):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        if last_val is None:
            last_val = np.zeros(self.num_rew_fns, dtype=np.float32)

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.vstack((self.rew_buf[path_slice], last_val))
        vals = np.vstack((self.val_buf[path_slice], last_val))
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        #TODO: see if there is a way to vectorize this
        for i in range(self.num_rew_fns):
            self.adv_buf[path_slice,i] = core.discount_cumsum(deltas[:,i], self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        # also store the cumulative returns for BROIL CVaR calculation
        self.posterior_returns.append(np.sum(rews, axis=0))
        
        self.path_start_idx = self.ptr

    def sample_cpc(self):
        assert self.ptr == self.max_size    # buffer has to be full before you can get self.ptr, 
        self.ptr, self.path_start_idx = 0, 0
        start = time.time()
        pos = self.obs_buf.copy()

        self.obs_buf = curl.utils.random_crop(self.obs_buf, self.image_size)
        pos = curl.utils.random_crop(pos, self.image_size)
    

        # the next two lines implement the advantage normalization trick
        #TODO: see if we can vectorize this and figure out multithreading
        for i in range(self.num_rew_fns):
            adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf[:,i])
            self.adv_buf[:,i] = (self.adv_buf[:,i] - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                        adv=self.adv_buf, logp=self.logp_buf, p_returns=self.posterior_returns)
        cpc_kwargs = dict(obs_anchor=self.obs_buf, obs_pos=pos, time_anchor=None, time_pos=None)
        self.posterior_returns = [] # resetting
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}, cpc_kwargs

        # return obses, actions, rewards, next_obses, not_dones, cpc_kwargs


    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get self.ptr, 
        self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        #TODO: see if we can vectorize this and figure out multithreading
        for i in range(self.num_rew_fns):
            adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf[:,i])
            self.adv_buf[:,i] = (self.adv_buf[:,i] - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                        adv=self.adv_buf, logp=self.logp_buf, p_returns=self.posterior_returns)
        self.posterior_returns = []
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



def vpg(env_fn, reward_dist, broil_risk_metric='cvar', actor_critic=core.BROILActorCritic, ac_kwargs=dict(), render=False, seed=0, 
        steps_per_epoch=4000, epochs=50, broil_lambda=0.5, broil_alpha=0.95, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=10):
    """
    Vanilla Policy Gradient 
    (with GAE-Lambda for advantage estimation)
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================
            The ``act`` method behaves the same as ``step`` but only returns ``a``.
            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================
            The ``v`` module's forward call should accept a batch of observations
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to VPG.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.
        broil_lambda (float): amount to blend between maximizing expected return (1.0)
            and maximizing CVaR (0.0). Always between 0 and 1.
        broil_alpha (float): risk sensitivity in range [0,1) for computing alpha-CVaR
            higher alpha is more risk sensitive.
        gamma (float): Discount factor. (Always between 0 and 1.)
        pi_lr (float): Learning rate for policy optimizer.
        vf_lr (float): Learning rate for value function optimizer.
        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.
        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    #print('env', args.env)
    #print('env.action_space', env.action_space)
    #print('env.env.action_space', env.env.action_space)
    # print('env unwrapped action meanings', env.unwrapped.action_space())
    # obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    if args.encoder_type == 'pixel':
            obs_dim = (3*args.frame_stack, args.image_size, args.image_size)
            pre_aug_obs_dim = (3*args.frame_stack,args.pre_transform_image_size,args.pre_transform_image_size)
    else:
        obs_dim = env.observation_space.shape
        pre_aug_obs_dim = obs_dim
    # Create BROIL actor-critic module
    num_rew_fns = len(reward_dist.posterior) #len(reward_dist.get_reward_distribution(env,np.zeros(obs_dim)))
    ac = actor_critic(obs_dim, env.action_space, num_rew_fns, encoder_feature_dim=args.encoder_feature_dim, encoder_lr=args.encoder_lr, curl_latent_dim=args.curl_latent_dim, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = VPGBuffer(pre_aug_obs_dim, act_dim, num_rew_fns, local_steps_per_epoch, gamma, lam, image_size=args.image_size)

    #### compute BROIL policy gradient loss (robust version)
    def compute_broil_weights(batch_rets, weights):
        '''batch_returns: list of numpy arrays of size num_rollouts x num_reward_fns
           weights: list of weights, e.g. advantages, rewards to go, etc by reward function over all rollouts,
            size is num_rollouts*ave_rollout_length x num_reward_fns
        '''
        #inputs are lists of numpy arrays
        #need to compute BROIL weights for policy gradient and convert to pytorch

        #first find the expected on-policy return for current policy under each reward function in the posterior
        exp_batch_rets = np.mean(batch_rets.numpy(), axis=0)
        print(exp_batch_rets)
        posterior_reward_weights = reward_dist.posterior


        #calculate sigma and find either the conditional value at risk or entropic risk measure given the current policy
        if broil_risk_metric == "cvar":
            #Calculate policy gradient for conditional value at risk

            sigma, cvar = cvar_enumerate_pg(exp_batch_rets, posterior_reward_weights, broil_alpha)
            print("sigma = {}, cvar = {}".format(sigma, cvar))

            #compute BROIL policy gradient weights
            total_rollout_steps = len(weights)
            broil_weights = np.zeros(total_rollout_steps, dtype=np.float64)
            for i, prob_r in enumerate(posterior_reward_weights):
                if sigma > exp_batch_rets[i]:
                    w_r_i = broil_lambda + (1 - broil_lambda) / (1 - broil_alpha)
                else:
                    w_r_i = broil_lambda
                broil_weights += prob_r * w_r_i * np.array(weights)[:,i]


            return broil_weights,cvar

        elif broil_risk_metric == "erm":
            #calculate policy gradient for entropic risk measure
            erm = -1.0 / broil_alpha * np.log(np.dot(posterior_reward_weights, np.exp(-broil_alpha * exp_batch_rets)))

            #compute stable weighted soft-max
            exponents = -broil_alpha * exp_batch_rets
            z = exponents - max(exponents)
            numerators = np.exp(z)
            denominator = np.dot(numerators, posterior_reward_weights)
            softmax_probs = numerators / denominator

            #compute BROIL policy gradient weights for ERM
            total_rollout_steps = len(weights)
            erm_weights = broil_lambda * np.ones(len(posterior_reward_weights)) + (1-broil_lambda) * softmax_probs

            broil_weights = np.array(weights) * erm_weights
            broil_weights = np.dot(broil_weights, posterior_reward_weights)

            return broil_weights,erm


        else:
            print("Risk metric not implemented!")
            raise NotImplementedError
 

    # Set up function for computing VPG policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old, batch_returns = data['obs'], data['act'], data['adv'], data['logp'], data['p_returns']

        # Use advantage estimates to compute BROIL policy gradient weights
        broil_weights, risk = compute_broil_weights(batch_returns, adv)
        weights = torch.as_tensor(broil_weights, dtype=torch.float32)
        # Policy loss
        pi, logp = ac.pi(obs, act)
        loss_pi = -(logp * weights).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info, risk

    # Set up function for computing value loss for a particular reward function value estimator
    # def compute_loss_v(data, reward_index):
    #     obs, ret = data['obs'], data['ret'][:,i]
    #     return ((ac.v(obs) - ret)**2).mean()

    #TODO not sure if this is correct, need to inspect...
    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        mse = (ac.v(obs) - ret)**2
        return (mse).mean()    


    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    #TODO: see if we can get away with one adam optimizer for family of networks...
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    #vf_optimizers = [Adam(ac.v.v_nets[i].parameters(), lr=vf_lr) for i in range(num_rew_fns)]

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(step):
        data, cpc_kwargs = buf.sample_cpc()

        # Get loss and info values before update
        pi_l_old, pi_info_old, risk =  compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with a single step of gradient descent
        pi_optimizer.zero_grad()
        loss_pi, pi_info, risk = compute_loss_pi(data)
        loss_pi.backward()
        mpi_avg_grads(ac.pi)    # average grads across MPI processes
        pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()
        
        if step % args.critic_target_update_freq == 0:
            curl.utils.soft_update_params(
                ac.v.encoder, ac.v.encoder_momentum,
                args.encoder_tau
            )

        if step % args.cpc_update_freq == 0 and ac.encoder_type == 'pixel':
            obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
            ac.update_cpc(obs_anchor, obs_pos,cpc_kwargs, None, step) # passed in L=None


        # Log changes from update
        kl, ent = pi_info['kl'], pi_info_old['ent']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old), Risk=risk, ExpectedRet=np.dot(np.mean(data['p_returns'].numpy(), axis=0), reward_dist.posterior))

    def render():
        img = env.env.render(mode='ansi')
        img = img.transpose(2, 0, 1).copy()
        return img

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(state_and_image=True), 0, 0
    if args.encoder_type == 'pixel':
        o = curl.utils.center_crop_image(o, args.image_size)
    print('o', o)
    o_state, o_image = o


    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        first_rollout = True
        for t in range(local_steps_per_epoch):

            # new
            # note: Can copy curl.train.py: L228-251.

            # \new

            step = epoch * local_steps_per_epoch + t
            # new
            o_image_4d = o_image[None,...]
            a, v, logp = ac.step(torch.as_tensor(o_image_4d, dtype=torch.float32))
            # \new
            a = a[0] #  Probably not needed anymore
            # print('action chosen', a)
            next_o, r, d, _ = env.step(a, state_and_image=True) # should work like nothing is changed if we pass in state_and_image=False
            next_o_state, next_o_image = next_o
            #TODO: check this, but I think reward as function of next state makes most sense
            # if args.env == 'CartPole-v0':
            if args.env == 'cartpole':
                # print('next_o.shape', next_o.shape)
                # print('next_o', next_o)
                rew_dist = reward_dist.get_reward_distribution(next_o_state)
                # OLD: rew_dist = reward_dist.get_reward_distribution(next_o_state)
                # print('rew_dist', rew_dist)
            elif args.env == 'PointBot-v0':
                rew_dist = reward_dist.get_reward_distribution(env, next_o_state)
                # OLD rew_dist = reward_dist.get_reward_distribution(env, next_o_state)
            else:
                raise NotImplementedError("Unsupported Environment")
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o_image, a, rew_dist, v, logp)
            logger.store(VVals=v)
            
            # Update obs (critical!)
            o_image = next_o_image
            o_state = next_o_state
            # OLD o = next_o_image

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if first_rollout:
                time.sleep(0.01)
                # print("cart position", o_state[0])
                
                

            if terminal or epoch_ended:
                first_rollout = False
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    o_image_4d = o_image[None,...]
                    _, v, _ = ac.step(torch.as_tensor(o_image_4d, dtype=torch.float32))
                    buf.finish_path(v)
                else:
                    buf.finish_path()
                
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(state_and_image=True), 0, 0
                o_state, o_image = o 


        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        
        # Perform VPG update!
        update(step)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('Risk', average_only=True)
        logger.log_tabular('ExpectedRet', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
    #env.plot_entire_trajectory()

if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='cartpole')#'CartPole-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000) # we already have training_steps in curl
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--risk_metric', type=str, default='cvar', help='choice of risk metric, options are "cvar" or "erm"' )
    parser.add_argument('--policy_lr', type=float, default=1e-2, help="learning rate for policy")
    parser.add_argument('--broil_lambda', type=float, default=0.5, help="blending between risk and expected perf")
    parser.add_argument('--broil_alpha', type=float, default=0.95, help="risk sensitivity for cvar [0,1) or erm (0,inf)")
    # parser.add_argument('--data_dir', type=stf, default='data', help="directory to save data")

    # new
    # FROM CURL:
    # parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='balance')
    parser.add_argument('--pre_transform_image_size', default=100, type=int)

    parser.add_argument('--image_size', default=100, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    # parser.add_argument('--agent', default='curl_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float) # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2, type=int) # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--curl_latent_dim', default=128, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    # parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')

    parser.add_argument('--log_interval', default=100, type=int)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, seed=args.seed)


    if args.env == 'cartpole':
        reward_dist = CartPoleReward()
    elif args.env == 'PointBot-v0':
        reward_dist = PointBotReward()
    else:
        raise NotImplementedError("Unsupported Environment")

    # new
    def env_fn():
        # env = dmc2gym.make(
        #     domain_name=args.env,
        #     task_name=args.task_name,
        #     seed=args.seed,
        #     visualize_reward=False,
        #     from_pixels=(args.encoder_type == 'pixel'),
        #     height=args.pre_transform_image_size,
        #     width=args.pre_transform_image_size,
        #     frame_skip=args.action_repeat
        # )
        # env = gym.make(args.env)
        env = DMCWrapper(
            domain_name=args.env,
            task_name=args.task_name,
            task_kwargs={'random':1},
            visualize_reward=False,
            from_pixels=(args.encoder_type == 'pixel'),
            height=args.pre_transform_image_size,
            width=args.pre_transform_image_size,
            camera_id=0,
            frame_skip=args.action_repeat,
            environment_kwargs=None,
            channels_first=True
        )

        env.seed(args.seed)

        # stack several consecutive frames together
        if args.encoder_type == 'pixel':
            env = curl.utils.FrameStack(env, k=args.frame_stack)
        return env

    vpg(env_fn=env_fn, reward_dist=reward_dist, broil_risk_metric=args.risk_metric, broil_lambda=args.broil_lambda, broil_alpha=args.broil_alpha,
        actor_critic=core.BROILActorCritic, render=args.render,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        pi_lr=args.policy_lr,
        logger_kwargs=logger_kwargs)
