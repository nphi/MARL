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

# provide arguments through command line
parser = argparse.ArgumentParser(description='Train RL agent for Chase-Paint Task')
parser.add_argument('--dir-out', required=True, type=str, help='Full path to output directory')
parser.add_argument('--arena-file', required=True, type=str, help='Name of the arena file')
parser.add_argument('--callback-file', type=str, default='callbacks_v1j', help='Name of the callback file')
parser.add_argument('--model-file', type=str, default='simple_rnn_v2_3_2', help='name of model file')
parser.add_argument('--l2-curr', type=float, default=3, help='L2-reg on recurrence')
parser.add_argument('--l2-inp', type=float, default=0, help='L2-reg on input')
parser.add_argument('--kl-coeff', type=float, default=0.2, help='kl coefficient')
parser.add_argument('--clip-param', type=float, default=0.3, help='PPO clipping')
parser.add_argument('--num-workers', type=int, default=5, help='number of workers deployed')
parser.add_argument('--num-gpus', type=int, default=1, help='number of GPUs deployed')
parser.add_argument('--train-iter', type=int, default=2000, help='number of training iterations')
parser.add_argument('--checkpoint-freq', type=int, default=100, help='freq of checkpoints')

args = parser.parse_args()

now = datetime.datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print(dt_string)
print('--------------------------------')

# Initialize ray
if args.num_gpus is not None:
    ray.init(log_to_driver=False, num_gpus=1, include_dashboard=False)
else:
    ray.init(log_to_driver=False, num_gpus=0, include_dashboard=False)
print('Ray initialized!')

# Import arena
arena = importlib.import_module(args.arena_file)
e = getattr(arena, args.arena_file)
env = e()
print(f"Imported arena {args.arena_file}.")

# Import callbacks
cb = importlib.import_module(args.callback_file)
myCallback = getattr(cb, 'MyCallbacks')

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


# Define an agent->policy mapping function.
def policy_mapping_fn(agent_id: str) -> str:
    # Make sure agent ID is valid.
    assert agent_id in ["agent1", "agent2"], f"ERROR: invalid agent ID {agent_id}!"
    id = agent_id[-1]
    return f'policy{id}'


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
    "kl_coeff": args.kl_coeff,
    "clip_param": args.clip_param,
}

max_seq_len = 20

# config.
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

# create logger with ray tune
from ray.tune import CLIReporter
reporter = CLIReporter(max_progress_rows=10, max_report_frequency=300, infer_limit=8)
reporter.add_metric_column('policy_reward_mean')
# train with ray tune
from ray import tune

stop = {
    # Note that the keys used here can be anything present in the above `rllib_trainer.train()` output dict.
    "training_iteration": args.train_iter,
}

print(f"Training with tune for {args.train_iter} iterations, saving checkpoint every {args.checkpoint_freq}")

# Run a simple experiment until one of the stopping criteria is met.
analysis = tune.run(
    PPOTrainer,
    progress_reporter=reporter,
    local_dir=args.dir_out,
    config=config,
    stop=stop,
    verbose=1,
    checkpoint_at_end=True,  # ... create a checkpoint when done.
    checkpoint_freq=args.checkpoint_freq,  # ... create a checkpoint every 100 training iterations.
)

last_checkpoint = analysis.get_last_checkpoint()

print("Training Finished")
print(f"Checkpoints saved to {last_checkpoint}")
now = datetime.datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print(dt_string)
