from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np
import scipy.io as scio
import pdb

class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self,
                         *,
                         worker,
                         base_env,
                         policies,
                         episode: MultiAgentEpisode,
                         env_index,
                         **kwargs):
        # We will use the `MultiAgentEpisode` object being passed into
        # all episode-related callbacks. It comes with a user_data property (dict),
        # which we can write arbitrary data into.

        # At the end of an episode, we'll transfer that data into the `hist_data`, and `custom_metrics`
        # properties to make sure our custom data is displayed in TensorBoard.

        # The episode is starting:
        # Set per-episode agent2-blocks counter (how many times has agent2 blocked agent1?).
        episode.user_data["num_collisions"] = 0
        # Set per-episode agent2 new fields
        episode.user_data["r2_new_fields"] = 0
        # Set per-episode agent2 paint attempt
        episode.user_data["r2_paint"] = 0
        # Set per-episode agent2 approaches
        episode.user_data["approaches"] = 0
        # Set per-episode agent1 escape far
        episode.user_data["escapes_far"] = 0
        # Set per-episode agent1 escape near
        episode.user_data["escapes_near"] = 0
        # Set per-episode agent1 escape close
        episode.user_data["escapes_close"] = 0
        # Set per-episode agent1 new fields
        episode.user_data["r1_new_fields"] = 0
        # Set per-episode agent1 paint attempt
        episode.user_data["r1_paint"] = 0

    def on_episode_step(self,
                        *,
                        worker,
                        base_env,
                        episode: MultiAgentEpisode,
                        env_index,
                        **kwargs):
        # Get both rewards.
        # Get info about escape close
        ag1_event = episode.last_info_for("agent1")['events'][0]
        ag2_event = episode.last_info_for("agent2")['events'][0]

        if "escape_close" in ag1_event:
            episode.user_data["escapes_close"] += 1
        if "escape_near" in ag1_event:
            episode.user_data["escapes_near"] += 1
        if "escape_far" in ag1_event:
            episode.user_data["escapes_far"] += 1
        if "agent1_new_field" in ag1_event:
            episode.user_data["r1_new_fields"] += 1
        if "agent1_paint" in ag1_event:
            episode.user_data["r1_paint"] += 1
        if "approach" in ag2_event:
            episode.user_data["approaches"] += 1
        if "agent2_new_field" in ag2_event:
            episode.user_data["r2_new_fields"] += 1
        if "agent2_paint" in ag2_event:
            episode.user_data["r2_paint"] += 1
        if "collision" in ag2_event:
            episode.user_data["num_collisions"] += 1

    def on_episode_end(self,
                       *,
                       worker,
                       base_env,
                       policies,
                       episode: MultiAgentEpisode,
                       env_index,
                       **kwargs):
        # Episode is done:
        # Write scalar values (sum over rewards) to `custom_metrics` and
        # time-series data (rewards per time step) to `hist_data`.
        # Both will be visible then in TensorBoard.
        episode.custom_metrics["approaches"] = episode.user_data["approaches"]
        episode.custom_metrics["r2_new_fields"] = episode.user_data["r2_new_fields"]
        episode.custom_metrics["r2_paint"] = episode.user_data["r2_paint"]
        episode.custom_metrics["num_collisions"] = episode.user_data["num_collisions"]
        episode.custom_metrics["escapes_far"] = episode.user_data["escapes_far"]
        episode.custom_metrics["escapes_near"] = episode.user_data["escapes_near"]
        episode.custom_metrics["escapes_close"] = episode.user_data["escapes_close"]
        episode.custom_metrics["r1_new_fields"] = episode.user_data["r1_new_fields"]
        episode.custom_metrics["r1_paint"] = episode.user_data["r1_paint"]
