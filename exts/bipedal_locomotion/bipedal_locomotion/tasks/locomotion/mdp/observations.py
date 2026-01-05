from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, ContactSensor, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def robot_joint_torque(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint torque of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.applied_torque.to(device)


def robot_joint_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint acc of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.joint_acc.to(device)


def robot_feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """contact force of the robot feet"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    contact_force_tensor = contact_sensor.data.net_forces_w_history.to(device)
    return contact_force_tensor.view(contact_force_tensor.shape[0], -1)


def robot_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_mass.to(device)

class BaseCOMWrapper:
    def __init__(self):
        self.count = 0
        self.coms = None
        self.body_ids = None
        self.__name__ = "base_com"

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """Returns the center of mass positions of the specified bodies in the asset.

        Args:
            env: The environment instance
            asset_cfg: The asset configuration containing body IDs to track

        Returns:
            torch.Tensor: Center of mass positions in world frame
        """
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        if self.count <= 1:
            # Initialize body IDs on first call
            if asset_cfg.body_ids == slice(None):
                self.body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
            else:
                self.body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

            # Get initial COM positions
            self.coms = asset.root_physx_view.get_coms()[:, self.body_ids, :3].to(env.device).squeeze(1)

        # Update COMs for first few steps to ensure stability
        if self.count < 5:
            self.coms = asset.root_physx_view.get_coms()[:, self.body_ids, :3].to(env.device).squeeze(1)

        self.count += 1
        if self.coms is None:
            self.coms = asset.root_physx_view.get_coms()[:, self.body_ids, :3].to(env.device).squeeze(1)

        return self.coms


base_com = BaseCOMWrapper()

class BodyMassWrapper:
    def __init__(self):
        self.count = 0
        self.masses = None
        self.__name__ = "body_mass"

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """The mass of the body of the asset."""
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        if self.count <= 1:
            self.masses = asset.root_physx_view.get_masses().to(device=asset.device)

        self.count += 1
        # Return the mass of the base for all environments
        return self.masses[:, asset_cfg.body_ids]


body_mass = BodyMassWrapper()


class BodyMassRelWrapper:
    def __init__(self):
        self.count = 0
        self.masses = None
        self.default_masses = None
        self.__name__ = "body_mass_rel"

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """The mass of the body of the asset."""
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        if self.count <= 1:
            self.masses = asset.root_physx_view.get_masses().to(device=asset.device)
            self.default_masses = asset.data.default_mass.to(device=asset.device)

        self.count += 1
        # Return the mass of the base for all environments
        return self.masses[:, asset_cfg.body_ids] - self.default_masses[:, asset_cfg.body_ids]


body_mass_rel = BodyMassRelWrapper()

class RigidBodyMaterialsWrapper:
    def __init__(self):
        self.count = 0
        self.materials = None
        self.__name__ = "rigid_body_materials"

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """The mass of the body of the asset."""
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        if self.count <= 1:
            self.materials = asset.root_physx_view.get_material_properties().to(device=asset.device)
            self.num_shapes_per_body = []
            for link_path in asset.root_physx_view.link_paths[0]:
                link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)

            # sample material properties from the given ranges
            body_count = 0
            self.body_ids = []
            for body_ids, valid in enumerate(self.num_shapes_per_body):
                if valid:
                    if isinstance(asset_cfg.body_ids, slice):
                        asset_cfg.body_ids = list(range(len(asset_cfg.body_ids)))
                    if body_ids in asset_cfg.body_ids:
                        self.body_ids.append(body_count)
                    body_count += 1

        self.count += 1

        return self.materials[:, self.body_ids, :].flatten(start_dim=1)


rigid_body_materials = RigidBodyMaterialsWrapper()

def foot_clearance_flag(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg = None,
    foot_radius: float = 0.032,
    contact_threshold: float = 1.0,
    clearance_threshold: float = 0.05,
) -> torch.Tensor:
    """Get a binary flag indicating if a foot has clearance issues.

    This function checks if there are significant height differences in the scanned area
    around the foot when it's in contact with the ground.

    Args:
        env: The RL environment instance
        sensor_cfg: Configuration for the foot's height scanner
        contact_sensor_cfg: Configuration for contact sensor (optional)
        foot_radius: Radius of the robot's foot
        contact_threshold: Threshold to determine if foot is in contact
        clearance_threshold: Threshold to determine if there's a clearance issue

    Returns:
        torch.Tensor: Binary flag (0.0 or 1.0) for each environment. Shape: (num_envs, 1)
    """
    # Get the height scanner data
    scanner = env.scene.sensors[sensor_cfg.name]
    heights = scanner.data.ray_hits_w[..., 2]  # Shape: [num_envs, num_rays]

    # Handle invalid height readings (inf)
    heights = torch.nan_to_num(heights, nan=-0.1, posinf=-0.1, neginf=-0.1)

    # Get foot height from scanner position
    foot_height = scanner.data.pos_w[:, 2].unsqueeze(1) - foot_radius

    # Calculate height differences
    height_diffs = torch.abs(foot_height - heights)

    # Clip small height differences to zero
    height_diffs = torch.clamp_min(height_diffs - 0.01, min=0)

    # Check if foot is in contact (if contact sensor is provided)
    if contact_sensor_cfg is not None:
        contact_sensor = env.scene.sensors[contact_sensor_cfg.name]
        # Get foot index from body_ids
        foot_idx = 0  # Adjust based on actual foot index in body_ids
        if sensor_cfg.name == "right_foot_height_scanner":
            foot_idx = 1  # Assuming right foot is second in body_ids

        contact_forces = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids]
        foot_contact = torch.norm(contact_forces[:, foot_idx], dim=-1) > contact_threshold

        # Only check for clearance when foot is in contact
        height_diffs = torch.where(foot_contact.unsqueeze(-1), height_diffs, torch.zeros_like(height_diffs))

    # Sum the height differences
    sum_height_diffs = torch.sum(height_diffs, dim=-1, keepdim=True)

    # Return binary flag (1.0 if sum exceeds threshold, 0.0 otherwise)
    return (sum_height_diffs > clearance_threshold).float()

def robot_inertia(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """inertia of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    inertia_tensor = asset.data.default_inertia.to(device)
    return inertia_tensor.view(inertia_tensor.shape[0], -1)


def robot_joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint positions of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_pos.to(device)


def robot_joint_stiffness(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint stiffness of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_stiffness.to(device)


def robot_joint_damping(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint damping of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_damping.to(device)


def robot_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """pose of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w.to(device)


def robot_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """velocity of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_vel_w.to(device)


def robot_material_properties(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """material properties of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    material_tensor = asset.root_physx_view.get_material_properties().to(device)
    return material_tensor.view(material_tensor.shape[0], -1)


def robot_center_of_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """center of mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    com_tensor = asset.root_physx_view.get_coms().clone().to(device)
    return com_tensor.view(com_tensor.shape[0], -1)


def robot_contact_force(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The contact forces of the body."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    body_contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]

    return body_contact_force.reshape(body_contact_force.shape[0], -1)


def get_gait_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the current gait phase as observation.

    The gait phase is represented by [sin(phase), cos(phase)] to ensure continuity.
    The phase is calculated based on the episode length and gait frequency.

    Returns:
        torch.Tensor: The gait phase observation. Shape: (num_envs, 2).
    """
    # check if episode_length_buf is available
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, 2, device=env.device)

    # Get the gait command from command manager
    command_term = env.command_manager.get_term("gait_command")
    # Calculate gait indices based on episode length
    gait_indices = torch.remainder(env.episode_length_buf * env.step_dt * command_term.command[:, 0], 1.0)
    # Reshape gait_indices to (num_envs, 1)
    gait_indices = gait_indices.unsqueeze(-1)
    # Convert to sin/cos representation
    sin_phase = torch.sin(2 * torch.pi * gait_indices)
    cos_phase = torch.cos(2 * torch.pi * gait_indices)

    return torch.cat([sin_phase, cos_phase], dim=-1)


def get_gait_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Get the current gait command parameters as observation.

    Returns:
        torch.Tensor: The gait command parameters [frequency, offset, duration].
                     Shape: (num_envs, 3).
    """
    return env.command_manager.get_command(command_name)


def robot_base_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """pose of the robot base"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w.to(device)

def feet_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids].flatten(start_dim=1)

def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)

def joint_pos_rel_exclude_wheel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                wheel_joints_name: list[str] = ["wheel_[RL]_Joint"] 
                                ) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)

    asset: Articulation = env.scene[asset_cfg.name]
    wheel_joints_idx = asset.find_joints(wheel_joints_name)[0]
    all_joints_idx = range(asset.num_joints)
    pos_idx_exclude_wheel = [i for i in all_joints_idx if i not in wheel_joints_idx]
    return asset.data.joint_pos[:, pos_idx_exclude_wheel] - asset.data.default_joint_pos[:, pos_idx_exclude_wheel]
