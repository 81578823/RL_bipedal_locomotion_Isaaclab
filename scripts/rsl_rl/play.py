"""RSL-RL智能体检查点播放脚本 / Script to play a checkpoint of an RL agent from RSL-RL."""

"""首先启动Isaac Sim仿真器 / Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# 添加argparse参数 / Add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch
import numpy as np

from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg,DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg, export_mlp_as_onnx, export_policy_as_jit
# 输入设备，用于键盘/手柄读取 / Input interface for keyboard/gamepad polling
import carb
import omni.appwindow


def main():
    """使用RSL-RL智能体进行测试 / Play with RSL-RL agent."""
    # 解析配置 / Parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.seed = agent_cfg.seed

    # 指定日志实验目录 / Specify directory for logging experiments
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path
    log_dir = os.path.dirname(resume_path)

    # 创建isaac环境 / Create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    # load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

    # === 手柄输入准备 / Prepare gamepad input ===
    input_iface = carb.input.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard() if app_window is not None else None
    # 订阅键盘事件，维护按键状态 / Subscribe to keyboard events to track key states
    pressed_keys = set()
    keyboard_sub_id = None

    def on_keyboard_event(e):
        from carb.input import KeyboardEventType

        # 仅记录我们关心的按下/释放事件 / Track press/release events
        if e.type in (KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT):
            pressed_keys.add(e.input)
        elif e.type == KeyboardEventType.KEY_RELEASE:
            pressed_keys.discard(e.input)
        return True

    if keyboard is not None and hasattr(input_iface, "subscribe_to_keyboard_events"):
        keyboard_sub_id = input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    # 兼容不同版本 carb.input 接口：优先 get_gamepad(idx)，否则用 get_gamepad_guid(idx)
    gamepad_handle = None
    if hasattr(input_iface, "get_gamepad"):
        try:
            gamepad_handle = input_iface.get_gamepad(0)
        except TypeError:
            # 某些版本 get_gamepad 不接受参数；直接调用
            try:
                gamepad_handle = input_iface.get_gamepad()
            except Exception:
                gamepad_handle = None
    if gamepad_handle is None and hasattr(input_iface, "get_gamepad_guid"):
        try:
            gamepad_handle = input_iface.get_gamepad_guid(0)
        except Exception:
            gamepad_handle = None

    def apply_keyboard_commands(cmd_tensor, lin_scale=1.0, ang_scale=0.8):
        """
        读取键盘按键并覆盖 base_velocity 命令。
        - W/S: 前进/后退
        - A/D: 左/右平移
        - Q/E: 左/右旋转（偏航角速度）
        """
        if keyboard is None or keyboard_sub_id is None:
            return cmd_tensor

        def pressed(key_code):
            return key_code in pressed_keys

        vx = 0.0
        vy = 0.0
        wz = 0.0
        # 线速度：W 前、S 后；A 左（负）、D 右（正）
        if pressed(carb.input.KeyboardInput.W):
            vx += lin_scale
        if pressed(carb.input.KeyboardInput.S):
            vx -= lin_scale
        if pressed(carb.input.KeyboardInput.A):
            vy -= lin_scale
        if pressed(carb.input.KeyboardInput.D):
            vy += lin_scale
        # 角速度：Q 左转（负），E 右转（正）
        if pressed(carb.input.KeyboardInput.Q):
            wz -= ang_scale
        if pressed(carb.input.KeyboardInput.E):
            wz += ang_scale

        # 兼容命令维度：常见 (N,3)=[vx,vy,wz] 或 (N,4)=[vx,vy,wz,heading]
        cmd_tensor[:, 0] = vx
        if cmd_tensor.shape[1] > 1:
            cmd_tensor[:, 1] = vy
        if cmd_tensor.shape[1] > 2:
            cmd_tensor[:, 2] = wz
        if cmd_tensor.shape[1] > 3:
            cmd_tensor[:, 3] = 0.0  # heading 保持 0
        return cmd_tensor

    def apply_gamepad_commands(cmd_tensor, lin_scale=1.2, ang_scale=0.8):
        """
        从手柄读取输入并覆盖 base_velocity 命令。
        - 左摇杆 Y：前后速度（前正，取反是因为上推为负值）
        - 左摇杆 X：侧向速度（右正）
        - 右摇杆 X：偏航角速度（右正）
        """
        if gamepad_handle is None:
            return cmd_tensor

        # 获取手柄状态：新接口为 input_iface.get_gamepad_state(handle)，旧接口可能是 handle.get_state()
        state = None
        if hasattr(input_iface, "get_gamepad_state"):
            try:
                state = input_iface.get_gamepad_state(gamepad_handle)
            except Exception:
                state = None
        if state is None and hasattr(gamepad_handle, "get_state"):
            try:
                state = gamepad_handle.get_state()
            except Exception:
                state = None
        if state is None:
            return cmd_tensor

        def _axis(obj, names):
            """从状态对象中按候选名称取出轴值，不存在则返回0。"""
            for n in names:
                if hasattr(obj, n):
                    try:
                        return float(getattr(obj, n))
                    except Exception:
                        continue
            return 0.0

        # 手柄轴通常在 [-1, 1]，根据需要缩放到期望速度范围 / Scale raw axes to desired command ranges
        vx = -_axis(state, ("left_stick_y", "leftStickY", "left_y", "leftY", "ly")) * lin_scale
        vy = _axis(state, ("left_stick_x", "leftStickX", "left_x", "leftX", "lx")) * lin_scale
        wz = _axis(state, ("right_stick_x", "rightStickX", "right_x", "rightX", "rx")) * ang_scale
        cmd_tensor[:, 0] = vx
        if cmd_tensor.shape[1] > 1:
            cmd_tensor[:, 1] = vy
        if cmd_tensor.shape[1] > 2:
            cmd_tensor[:, 2] = wz
        # heading 保持 0，交给策略自行选择航向 / Keep heading zero; policy handles heading
        if cmd_tensor.shape[1] > 3:
            cmd_tensor[:, 3] = 0.0
        return cmd_tensor

     # 导出策略到onnx / Export policy to onnx
    if EXPORT_POLICY:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.actor_critic, export_model_dir
        )
        print("Exported policy as jit script to: ", export_model_dir)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor, 
            export_model_dir, 
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            export_model_dir,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )
    # reset environment
    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict["observations"].get("commands") 
    # 记录命令与实际速度（限制记录步数防止占用过多内存） / buffers for cmd vs actual velocity
    log_cmd = []
    log_vel = []
    log_max = 800
    # simulate environment
    maxstep=800
    step = 0
    while simulation_app.is_running() and step < maxstep:
        step += 1
        # run everything in inference mode
        with torch.inference_mode():
            # 读取手柄并更新命令：直接写入 env 的 base_velocity 命令缓冲 / Poll gamepad to drive base_velocity
            if "commands" in obs_dict["observations"]:
                # 优先直接修改 command_term 的 buffer，避免缺少 set_command 接口 / mutate term buffer directly
                cmd_term = None
                if hasattr(env.unwrapped, "command_manager"):
                    try:
                        cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
                    except Exception:
                        cmd_term = None
                # commands 张量形状 (num_envs, 4): [lin_x, lin_y, ang_z, heading]
                if cmd_term is not None and hasattr(cmd_term, "command"):
                    # 先键盘、后手柄：键盘存在则覆盖，否则尝试手柄 / keyboard first, then gamepad
                    cmd_term.command[:] = apply_keyboard_commands(cmd_term.command)
                    cmd_term.command[:] = apply_gamepad_commands(cmd_term.command)
                    commands = cmd_term.command
                else:
                    # 回退仅供策略输入使用，不会影响环境内部命令 / fallback only affects policy input
                    commands = apply_keyboard_commands(commands)
                    commands = apply_gamepad_commands(commands)

            # agent stepping
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            # env stepping
            obs, _, _, infos = env.step(actions)
            obs_history = infos["observations"].get("obsHistory")
            obs_history = obs_history.flatten(start_dim=1)
            commands = infos["observations"].get("commands") 
            obs_dict = infos  # 下一帧继续使用最新观测 / keep newest obs_dict for next loop
            # 记录当前命令与实际速度（取第一个 env）/ log commanded vs actual base velocity
            if len(log_cmd) < log_max and isinstance(commands, torch.Tensor):
                try:
                    cmd_np = commands[0].detach().cpu().numpy()
                    robot = env.unwrapped.scene["robot"]
                    base_vel = robot.data.root_lin_vel_w[:, :2]
                    vel_np = base_vel[0].detach().cpu().numpy()
                    log_cmd.append(cmd_np[:2])
                    log_vel.append(vel_np)
                except Exception:
                    pass

    # close the simulator
    env.close()

    # 保存命令与实际速度曲线 / save plot of command vs velocity
    if len(log_cmd) > 0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.arange(len(log_cmd)) * env.unwrapped.step_dt
        cmd_arr = np.vstack(log_cmd)
        vel_arr = np.vstack(log_vel)

        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(t, cmd_arr[:, 0], label="cmd vx")
        axes[0].plot(t, vel_arr[:, 0], label="actual vx")
        axes[0].set_ylabel("vx (m/s)")
        axes[0].legend()

        axes[1].plot(t, cmd_arr[:, 1], label="cmd vy")
        axes[1].plot(t, vel_arr[:, 1], label="actual vy")
        axes[1].set_ylabel("vy (m/s)")
        axes[1].set_xlabel("time (s)")
        axes[1].legend()

        fig.tight_layout()
        plot_path = os.path.join(log_dir, "play_cmd_vs_vel.png")
        fig.savefig(plot_path)
        print(f"[INFO] Saved command vs velocity plot to {plot_path}")


if __name__ == "__main__":
    EXPORT_POLICY = True
    # run the main execution
    main()
    # close sim app
    simulation_app.close()