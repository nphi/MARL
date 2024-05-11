# Import libraries
import argparse
import datetime
import os
import sys
import importlib

import ray
import gym
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTorchPolicy
import pprint
import numpy as np

from HelperFunction import renderVideo

# provide arguments through command line
parser = argparse.ArgumentParser(description='Train RL agent for Chase-Paint Task')
parser.add_argument('--dir-in', required=True, type=str, help='Full path to checkpoint directory')
parser.add_argument('--dir-out',required=True, type=str, help='Full path to out directory')
parser.add_argument('--arena-file', required=True, type=str, help='Name of the arena file')
parser.add_argument('--num-checkpoints', type=int, default=20, help='number of checkpoints')
parser.add_argument('--timestep-limit', type=int, default=500, help='number of timesteps per episode')
parser.add_argument('--num-episodes', type=int, default=50, help='number of timesteps per episode')
parser.add_argument('--num-workers', type=int, default=0, help='number of workers deployed')
parser.add_argument('--num-gpus', type=int, default=0, help='number of GPUs deployed')
parser.add_argument('--callback-file', type=str, default='callbacks_v1j', help='Name of the callback file')
parser.add_argument('--model-file', type=str, default='simple_rnn_v2_3_2', help='name of model file')
parser.add_argument('--l2-curr', type=int, default=3, help='L2-reg on recurrence')
parser.add_argument('--l2-inp', type=int, default=0, help='L2-reg on input')
parser.add_argument('--video-out', type=bool, default=False, help='video output set to False')
parser.add_argument('--fps', type=int, default=2, help='video fps')
parser.add_argument('--video-path', type=str, help='video output path')
parser.add_argument('--file-suffix', type=str, help='suffix of output file name')
args = parser.parse_args()

# Print datetime for logging
now = datetime.datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print(dt_string)
print('--------------------------------')

# Initialize ray
ray.init(log_to_driver=False, num_gpus=0, include_dashboard=False)
print('Ray initialized!')

# Import arena
arena = importlib.import_module(args.arena_file)
e = getattr(arena, args.arena_file)
env = e()
print(f"Imported arena {args.arena_file}.")

# Import training model & register them
md = importlib.import_module(args.model_file)
if args.model_file == "simple_fc_net":
    myModel = getattr(md, 'CustomFCNet')
    modelName = "FC"
    ModelCatalog.register_custom_model(modelName, myModel)
else:
    myModel = getattr(md, 'AnotherTorchRNNModel')
    modelName = 'rnn_noFC'
    ModelCatalog.register_custom_model(modelName, myModel)
print(f"Imported model {args.model_file}.")

policies = {
    "policy1": (PPOTorchPolicy, env.observation_space, env.action_space, {}),
    "policy2": (PPOTorchPolicy, env.observation_space, env.action_space, {}),
}

# Import callbacks
cb = importlib.import_module(args.callback_file)
myCallback = getattr(cb, 'MyCallbacks')

# Define an agent->policy mapping function.
def policy_mapping_fn(agent_id: str) -> str:
    # Make sure agent ID is valid.
    assert agent_id in ["agent1", "agent2"], f"ERROR: invalid agent ID {agent_id}!"
    id = agent_id[-1]
    return f'policy{id}'

# Set Config
config = {
    "env": e,
    "callbacks": myCallback,
    "env_config": {
        "config": {
            "width": 10,
            "height": 10,
            "ts": 100,
        },
    },

    "framework": "torch",  # If users have chosen to install torch instead of tf.
    "grad_clip": None,
}

max_seq_len = 20

# RNN config.
config.update({
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
    },
    "model": {
        "custom_model": modelName,
        "max_seq_len": max_seq_len,
        "custom_model_config": {
            "fc_size": 200,
            "rnn_hidden_size": 256,
            # "device": torch.device("cuda:0"),
            "l2_lambda": args.l2_curr,
            "l2_lambda_inp": args.l2_inp,
        },
    },
    "num_workers": args.num_workers,
    "num_gpus": args.num_gpus,
    "_fake_gpus": False,
})
pprint.pprint(config)

## Get checkpoint locations
# set check point location
dirName = args.dir_in
import os
checkpoint_file = dirName+"\\checkpoint-000100"
filePresent = os.listdir(os.path.dirname(checkpoint_file))
iterList = []

# # retrieve file index first
for fileIdx in range(args.num_checkpoints):
    # filePresent[fileIdx]
    iterList.append(filePresent[fileIdx])

# recombine fileName
checkpoint_list = []
for fileIdx in range(args.num_checkpoints):
    splitComp = iterList[fileIdx].split('_')
    newComp = splitComp[0] + '-' + splitComp[1].lstrip('0')
    checkpoint_list.append((dirName + '\\' + iterList[fileIdx]))  #+'\\'+newComp))b
print("Checkpoint directory acquired")

## Rnn specific roll-out
rllib_trainer = PPOTrainer(config)
import torch
import scipy.io as scio

# Restore trainer at different checkpoints and run rolled-out episodes
for iterIdx in range(args.num_checkpoints):
    if iterIdx > 0:
        rllib_trainer.restore(checkpoint_list[iterIdx])
    # Retrieve evaluation episodes with information on position, distance, observation space,
    model_1 = rllib_trainer.get_policy('policy1').model
    model_2 = rllib_trainer.get_policy('policy2').model

    print(f'Producing roll-outs for checkpoint{rllib_trainer.iteration}')
    for t in range(args.num_episodes):
        frame_dict = dict()
        bv_dict = dict()

        # Clear activations
        model_1.activations = {}
        model_2.activations = {}

        # Disable, then enable activation saving.
        model_1.deregister_activation_hooks()
        model_1.register_activation_hooks()

        model_2.deregister_activation_hooks()
        model_2.register_activation_hooks()

        # render environment
        env.timestep_limit = args.timestep_limit
        obs = env.reset()

        timelength = env.timestep_limit

        # prelocate matrix for bv dict
        pos_1 = np.zeros((timelength,2))
        pos_2 = np.zeros((timelength,2))
        d     = np.zeros((timelength,1))
        obsAll = []
        eventAll = []

        # initialize state for the first step
        init_state = model_1.get_initial_state()
        state_list1 = init_state
        state_list2 = init_state

        count = 0

        a1_list = []
        a2_list = []

        while True:
            # compute action
            a1,state_out1,_ = rllib_trainer.compute_single_action(obs["agent1"], state = state_list1, policy_id="policy1")
            a2,state_out2,_ = rllib_trainer.compute_single_action(obs["agent2"], state = state_list2, policy_id="policy2")
            obs, rewards, dones, events = env.step({"agent1": a1, "agent2": a2})

            # retrieve position, distance, events and observation space
            pos_1[env.timesteps-1,:]=env.agent1_pos
            pos_2[env.timesteps-1,:]=env.agent2_pos
            d[env.timesteps-1,:]    = np.linalg.norm(np.array((env.agent1_pos)-np.array(env.agent2_pos)))
            eventAll.append(events)
            obsAll.append(obs)
            a1_list.append(a1)
            a2_list.append(a2)

            # get corresponding weight
            input_weight1 = model_1.rnn.cpu().weight_ih_l0.detach().numpy()
            hidden_weight1  =  model_1.rnn.cpu().weight_hh_l0.detach().numpy()
            out_weight1     = model_1.action_branch.cpu().weight.detach().numpy()

            input_weight2 = model_2.rnn.cpu().weight_ih_l0.detach().numpy()
            hidden_weight2  =  model_2.rnn.cpu().weight_hh_l0.detach().numpy()
            out_weight2     = model_2.action_branch.cpu().weight.detach().numpy()

            frame_dict[env.timesteps-1] = env.render_to_image()
            count += 1
            if dones["__all__"]==True:
                break
            else:
                state_list1 = state_out1[:]
                state_list2 = state_out2[:]

        # retrieve activity from hook
        x1 = np.zeros((timelength,256),dtype=np.float32)
        x2 = np.zeros((timelength,256),dtype=np.float32)
        bv1 = np.zeros((timelength,8),dtype=int)
        bv2 = np.zeros((timelength,8),dtype=int)
        a1_mat = np.zeros((timelength,5),dtype=int)
        a2_mat = np.zeros((timelength,5),dtype=int)

        for j in range(timelength):
            x1[j,:] = np.array(model_1.activations['rnn'][j][0][0])
            x2[j,:] = np.array(model_2.activations['rnn'][j][0][0])
            a1_mat[j,a1_list[j]] = 1
            a2_mat[j,a2_list[j]] = 1

        for i in range(timelength):
            if 'agent2_new_field' in eventAll[i]['agent2']['events'][0]:
                bv2[i,1]=1
            if 'collision' in eventAll[i]['agent2']['events'][0]:
                bv2[i,2]=1
            elif 'approach' in eventAll[i]['agent2']['events'][0]:
                bv2[i,3]=1
            elif 'agent2_paint' in eventAll[i]['agent2']['events'][0]:
                bv2[i,7]=1
            else:
                bv2[i,0]=1

            if 'agent1_new_field' in eventAll[i]['agent1']['events'][0]:
                bv1[i,1]=1
            if 'collision' in eventAll[i]['agent1']['events'][0]:
                bv1[i,2]=1
            elif 'escape_far' in eventAll[i]['agent1']['events'][0]:
                bv1[i,4]=1
            elif 'escape_near' in eventAll[i]['agent1']['events'][0]:
                bv1[i,5]=1
            elif 'escape_close' in eventAll[i]['agent1']['events'][0]:
                bv1[i,6]=1
            elif 'agent1_paint' in eventAll[i]['agent1']['events'][0]:
                bv1[i,7]=1
            else:
                bv1[i,0]=1

        bv_dict['R1'] = x1
        bv_dict['R2'] = x2
        bv_dict['Bv1'] = bv1
        bv_dict['Bv2'] = bv2
        bv_dict['Pos_R1']= pos_1
        bv_dict['Pos_R2']= pos_2
        bv_dict['Distance'] = d
        bv_dict['Obs'] = obsAll
        bv_dict['A1']  = a1_mat
        bv_dict['A2']  = a2_mat
        bv_dict['action1']  = a1_list
        bv_dict['action2']  = a2_list
        bv_dict['W_inp1']  = np.array(input_weight1.T)
        bv_dict['W_inp2']  = np.array(input_weight2.T)
        bv_dict['W_rec1']  = np.array(hidden_weight1.T)
        bv_dict['W_rec2']  = np.array(hidden_weight2.T)
        bv_dict['W_out1']  = np.array(out_weight1.T)
        bv_dict['W_out2']  = np.array(out_weight2.T)

        dirOut = args.dir_out
        title = 'behavior_output_' + str(rllib_trainer.iteration)+'iters_test_'+str(t+1)+args.file_suffice+'.mat'
        scio.savemat(dirOut+title,bv_dict)
        if (args.video_out == True) and (t<5):
            outName = args.video_path+"video_"+ str(rllib_trainer.iteration)+'iters_'+str(t+1)
            renderVideo(outName,frameDict = frame_dict,fps=args.fps)
            print('Video rendered successfully')

print("Evaluation Finished")
print(f"out_files saved to {args.dir_out}")
now = datetime.datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print(dt_string)
