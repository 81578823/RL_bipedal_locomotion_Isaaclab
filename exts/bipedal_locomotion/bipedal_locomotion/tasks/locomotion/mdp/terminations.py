from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

"""
MDP terminations.
"""


def action_out_of_limits(
    env: ManagerBasedRLEnv, threshold: float = 5.0
) -> torch.Tensor:
    """Terminate when the action values exceed the specified threshold.

    This function checks if any action value exceeds the threshold limit,
    which can help prevent training instability caused by extremely large actions.

    Args:
        env: The environment instance.
        threshold: Maximum allowed absolute value for actions. Default is 5.0.

    Returns:
        Boolean tensor indicating which environments should be terminated.
    """
    # Get the current actions from the environment
    actions = env.action_manager.action

    # Check if any action exceeds the threshold (in absolute value)
    action_magnitude = torch.abs(actions)
    max_action_per_env = torch.max(action_magnitude, dim=1)[0]

    # Return True for environments where max action exceeds threshold
    return max_action_per_env > threshold


class TimeOutStochasticWrapper:
    """
    A wrapper class for calculating stochastic time_out condition.
    """

    def __init__(self):
        self.max_episode_length = None
        self.__name__ = "time_out_stochastic"

    def __call__(self, env: ManagerBasedRLEnv, rand_scale: float) -> torch.Tensor:
        if self.max_episode_length is None:
            self.max_episode_length = (
                torch.ones(env.episode_length_buf.shape, device=env.episode_length_buf.device) * env.max_episode_length
            )
            random_values = torch.rand(self.max_episode_length.shape, device=self.max_episode_length.device)
            random_scale = (random_values - 0.5) * rand_scale + 1
            self.max_episode_length *= random_scale

        time_out = env.episode_length_buf >= self.max_episode_length

        # resample max_episode_length with stochastic mask
        time_out_ids = time_out.nonzero(as_tuple=False).squeeze(-1)
        random_values = torch.rand(time_out_ids.shape, device=time_out_ids.device)
        mask = random_values < 0.1
        time_out_ids_ = time_out_ids[mask]
        random_values = torch.rand(time_out_ids_.shape, device=time_out_ids_.device)
        random_scale = (random_values - 0.5) * rand_scale + 1
        self.max_episode_length[time_out_ids_] = env.max_episode_length * random_scale

        return time_out


time_out_stochastic = TimeOutStochasticWrapper()


"""
Fail terminations.
"""


def bad_orientation_stochastic(
    env: ManagerBasedRLEnv,
    roll_limit_angle: float,
    pitch_limit_angle: float,
    probability: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits."""
    # extract the used quantities
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity_vector = asset.data.projected_gravity_b

    pitch = torch.atan2(gravity_vector[:, 0], torch.sqrt(gravity_vector[:, 1] ** 2 + gravity_vector[:, 2] ** 2))
    roll = torch.atan2(gravity_vector[:, 1], torch.sqrt(gravity_vector[:, 0] ** 2 + gravity_vector[:, 2] ** 2))

    # Check if orientation exceeds limits - using element-wise operations
    bad_roll = torch.abs(roll) > roll_limit_angle
    bad_pitch = torch.abs(pitch) > pitch_limit_angle
    bad_orientation = torch.logical_or(bad_roll, bad_pitch)

    # Apply stochastic termination
    random_values = torch.rand(bad_orientation.shape, device=bad_orientation.device)
    return bad_orientation & (random_values < probability)


def illegal_contact_stochastic(
    env: ManagerBasedRLEnv, threshold: float, probability: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    illegal_contact = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )
    random_values = torch.rand(illegal_contact.shape, device=illegal_contact.device)
    return illegal_contact & (random_values < probability)


def bad_torque_stochastic(
    env: ManagerBasedRLEnv,
    limit_torque: float,
    max_duration: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the torque on the asset exceeds the torque threshold."""

    if not hasattr(env, "high_torque_duration"):
        env.high_torque_duration = torch.zeros(env.num_envs, device=env.device)

    # Extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    # Check if any joint torque exceeds or equals the limit
    current_high_torque = torch.any(asset.data.applied_torque[:, asset_cfg.joint_ids] >= limit_torque, dim=-1)

    # Update high torque duration counter
    env.high_torque_duration = torch.where(
        current_high_torque, env.high_torque_duration + 1, torch.zeros_like(env.high_torque_duration)
    )

    # Determine which environments should terminate
    should_terminate = env.high_torque_duration >= max_duration

    return should_terminate
