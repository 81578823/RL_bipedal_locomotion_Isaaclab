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
import math

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

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 最大步数固定 800：录像、绘图、仿真循环统一使用， 这里 100step对应 25s
    max_steps = 400

    # Wrap video capture after any MARL -> single-agent conversion to avoid losing the wrapper
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": max_steps,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

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

    def _resolve_key_codes(*names):
        """Return available KeyboardInput enums from candidate attribute names."""
        codes = []
        for name in names:
            code = getattr(carb.input.KeyboardInput, name, None)
            if code is not None:
                codes.append(code)
        return tuple(codes)

    # 数字键（主键盘和小键盘）候选枚举名称 / Candidate enum names for number keys
    DIGIT_KEYS = {
        "1": _resolve_key_codes("_1", "KEY_1", "ONE", "KP_1", "NUMPAD_1"),
        "2": _resolve_key_codes("_2", "KEY_2", "TWO", "KP_2", "NUMPAD_2"),
        "3": _resolve_key_codes("_3", "KEY_3", "THREE", "KP_3", "NUMPAD_3"),
        "4": _resolve_key_codes("_4", "KEY_4", "FOUR", "KP_4", "NUMPAD_4"),
    }

    def _get_step_dt_seconds():
        """Estimate simulated seconds per env.step for cooldown calculation."""
        sim_dt = None
        decimation = 1
        try:
            sim_dt = float(getattr(env.unwrapped.sim, "dt", 0.0))
        except Exception:
            sim_dt = None
        try:
            decimation = float(getattr(env.unwrapped, "decimation", 1))
        except Exception:
            decimation = 1
        if sim_dt is None or sim_dt <= 0.0:
            sim_dt = 0.02
        return max(sim_dt * decimation, 1e-4)

    def _get_base_body_ids(robot):
        """Resolve base body ids; fallback to first body if name lookup fails."""
        def _to_int_ids(values):
            ids = []
            for v in values:
                if isinstance(v, (int, np.integer)):
                    ids.append(int(v))
                elif isinstance(v, str):
                    try:
                        if hasattr(robot, "body_names") and v in robot.body_names:
                            ids.append(int(robot.body_names.index(v)))
                    except Exception:
                        continue
            return ids

        if hasattr(robot, "find_bodies"):
            try:
                ids = _to_int_ids(robot.find_bodies("base_Link"))
                if ids:
                    return ids
            except Exception:
                pass
        try:
            if hasattr(robot, "body_names") and "base_Link" in robot.body_names:
                return [int(robot.body_names.index("base_Link"))]
        except Exception:
            pass
        return [0]

    def _apply_body_push(robot, force_body, body_ids=None):
        """Apply a one-frame body-frame force to all envs on selected bodies."""
        if robot is None or not hasattr(robot, "set_external_force_and_torque"):
            return
        device = robot.device
        num_envs = robot.num_instances if hasattr(robot, "num_instances") else env.unwrapped.scene.num_envs
        env_ids = torch.arange(num_envs, device=device)
        target_body_ids = body_ids if body_ids is not None else [0]
        base_force_buffer = getattr(robot, "_external_force_b", None)
        dtype = base_force_buffer.dtype if base_force_buffer is not None else torch.float32
        force_tensor = torch.tensor(force_body, device=device, dtype=dtype)
        force_tensor = force_tensor.view(1, 1, 3).expand(len(env_ids), len(target_body_ids), 3).clone()
        torque_tensor = torch.zeros_like(force_tensor)
        # Clear any previous external force/torque to avoid accumulation
        if base_force_buffer is not None:
            robot._external_force_b *= 0
        if hasattr(robot, "_external_torque_b"):
            robot._external_torque_b *= 0
        robot.set_external_force_and_torque(force_tensor, torque_tensor, env_ids=env_ids, body_ids=target_body_ids)

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
    # 禁用随机推搡事件，仅在play阶段 / Disable stochastic push during play only
    if hasattr(env.unwrapped, "event_manager"):
        try:
            term_cfg = env.unwrapped.event_manager.get_term_cfg("push_robot")
            if hasattr(term_cfg, "params") and isinstance(term_cfg.params, dict):
                term_cfg.params["probability"] = 0.0
            env.unwrapped.event_manager.set_term_cfg("push_robot", term_cfg)
            print("[INFO] Disabled stochastic push_robot event during play.")
        except Exception:
            pass
    # 手动push参数 / Manual push settings
    push_force_templates = [
        {"keys": DIGIT_KEYS["1"], "force": (1600.0, 0.0, 0.0), "desc": "forward"},
        {"keys": DIGIT_KEYS["2"], "force": (-1400.0, 0.0, 0.0), "desc": "backward"},
        {"keys": DIGIT_KEYS["3"], "force": (0.0, 1200.0, 0.0), "desc": "left"},
        {"keys": DIGIT_KEYS["4"], "force": (0.0, -1000.0, 0.0), "desc": "right"},
    ]
    push_cooldown_steps = max(1, int(math.ceil(0.5 / _get_step_dt_seconds())))
    last_push_step = -push_cooldown_steps
    robot_asset = env.unwrapped.scene["robot"]
    push_body_ids = _get_base_body_ids(robot_asset)
    # 记录命令与实际速度（限制记录步数防止占用过多内存） / buffers for cmd vs actual velocity
    log_cmd = []
    log_vel = []
    log_ang_cmd = []
    log_ang_vel = []
    log_max = max_steps
    log_counter = 0
    
    # 获取命令接口
    cmd_term = None
    if hasattr(env.unwrapped, "command_manager"):
        try:
            cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        except Exception:
            cmd_term = None
    
    # simulate environment
    maxstep = max_steps
    step = 0
    while simulation_app.is_running() and step < maxstep:
        if step % 100 == 0:
            print(f"[DEBUG] step={step}, maxstep={maxstep}, sim_running={simulation_app.is_running()}")
        step += 1
        # run everything in inference mode
        with torch.inference_mode():
            # 读取手柄并更新命令：直接写入 env 的 base_velocity 命令缓冲 / Poll gamepad to drive base_velocity
            if "commands" in obs_dict["observations"]:
                # 先获取当前命令
                current_commands = commands.clone() if isinstance(commands, torch.Tensor) else commands.copy()
                
                # 应用键盘/手柄输入
                if cmd_term is not None and hasattr(cmd_term, "command"):
                    # 记录应用前的命令
                    cmd_before = cmd_term.command.clone()
                    # 应用键盘/手柄输入
                    cmd_term.command[:] = apply_keyboard_commands(cmd_term.command)
                    cmd_term.command[:] = apply_gamepad_commands(cmd_term.command)
                    # 记录应用后的命令
                    commands = cmd_term.command.clone()
                else:
                    # 回退方案
                    commands = apply_keyboard_commands(commands)
                    commands = apply_gamepad_commands(commands)
                
                # 记录命令
                if log_counter < log_max and isinstance(commands, torch.Tensor):
                    try:
                        cmd_np = commands[0].detach().cpu().numpy()
                        
                        # 记录命令
                        log_cmd.append(cmd_np[:2])  # vx, vy
                        if cmd_np.shape[0] > 2:
                            log_ang_cmd.append(cmd_np[2])  # wz
                        else:
                            log_ang_cmd.append(0.0)
                        
                        log_counter += 1
                    except Exception as e:
                        print(f"[ERROR] Failed to log command at step {step}: {e}")
                        pass

            # 处理自定义push（相对机器人朝向，0.5s冷却） / Handle custom pushes (body-frame, 0.5s cooldown)
            if keyboard is not None and keyboard_sub_id is not None:
                if (step - last_push_step) >= push_cooldown_steps:
                    triggered = None
                    for tpl in push_force_templates:
                        if any(code in pressed_keys for code in tpl["keys"]):
                            triggered = tpl
                            break
                    if triggered is not None and len(triggered["keys"]) > 0:
                        _apply_body_push(robot_asset, triggered["force"], body_ids=push_body_ids)
                        last_push_step = step

            # agent stepping
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            # env stepping
            obs, _, _, infos = env.step(actions)
            obs_history = infos["observations"].get("obsHistory")
            obs_history = obs_history.flatten(start_dim=1)
            commands = infos["observations"].get("commands") 
            obs_dict = infos  # 下一帧继续使用最新观测 / keep newest obs_dict for next loop
            
            # 在步进后记录实际速度
            if len(log_vel) < len(log_cmd):  # 确保与命令对应
                try:
                    robot = env.unwrapped.scene["robot"]
                    
                    # 获取机体坐标系下的实际速度
                    if hasattr(robot.data, 'root_lin_vel_b'):
                        # 如果机器人数据直接提供机体坐标系速度
                        base_vel_b = robot.data.root_lin_vel_b[0, :2]
                        base_ang_vel_b = robot.data.root_ang_vel_b[0, 2]  # 假设z轴是偏航
                    else:
                        # 否则从世界坐标系转换到机体坐标系
                        from isaaclab.utils.math import quat_rotate, quat_inv
                        quat = robot.data.root_quat_w[0]  # 四元数
                        base_vel_w = robot.data.root_lin_vel_w[0, :3]
                        base_ang_vel_w = robot.data.root_ang_vel_w[0, :3]
                        
                        # 将世界坐标系速度转换到机体坐标系
                        base_vel_b = quat_rotate(quat_inv(quat), base_vel_w)
                        base_ang_vel_b = quat_rotate(quat_inv(quat), base_ang_vel_w)
                        base_ang_vel_b = base_ang_vel_b[2]  # 只取z轴分量
                    
                    vel_np = base_vel_b.detach().cpu().numpy()
                    ang_vel_np = base_ang_vel_b.detach().cpu().item() if hasattr(base_ang_vel_b, 'item') else float(base_ang_vel_b)
                    
                    log_vel.append(vel_np)
                    log_ang_vel.append(ang_vel_np)
                    
                    # 调试输出
                    if step % 50 == 0 and len(log_cmd) > 0 and len(log_vel) > 0:
                        idx = len(log_vel) - 1
                        print(f"[DEBUG] Step {step}: "
                              f"Cmd=[{log_cmd[idx][0]:.2f}, {log_cmd[idx][1]:.2f}, {log_ang_cmd[idx]:.2f}], "
                              f"Vel=[{log_vel[idx][0]:.2f}, {log_vel[idx][1]:.2f}, {log_ang_vel[idx]:.2f}]")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to log velocity at step {step}: {e}")
                    # 添加占位符
                    if len(log_vel) < len(log_cmd):
                        log_vel.append(np.array([0.0, 0.0]))
                        log_ang_vel.append(0.0)

    print(f"[INFO] Exiting play loop at step {step}")

    # close the simulator
    env.close()

    # 保存命令与实际速度曲线 / save plot of command vs velocity
    # 确保命令和速度数组长度相同
    min_len = min(len(log_cmd), len(log_vel))
    if min_len > 0:
        log_cmd = log_cmd[:min_len]
        log_vel = log_vel[:min_len]
        log_ang_cmd = log_ang_cmd[:min_len]
        log_ang_vel = log_ang_vel[:min_len]
        
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = np.arange(min_len)
        cmd_arr = np.vstack(log_cmd)
        vel_arr = np.vstack(log_vel)
        ang_cmd_arr = np.array(log_ang_cmd)
        ang_vel_arr = np.array(log_ang_vel)

        # 均方误差 / mean squared error for each channel
        mse_vx = float(np.mean((cmd_arr[:, 0] - vel_arr[:, 0]) ** 2))
        mse_vy = float(np.mean((cmd_arr[:, 1] - vel_arr[:, 1]) ** 2))
        mse_wz = float(np.mean((ang_cmd_arr - ang_vel_arr) ** 2))

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        # 绘制x方向速度
        axes[0].plot(steps, cmd_arr[:, 0], 'b-', linewidth=1.5, alpha=0.7, label="Command vx")
        axes[0].plot(steps, vel_arr[:, 0], 'r-', linewidth=1.5, alpha=0.7, label="Actual vx")
        axes[0].set_ylabel("vx (m/s)")
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].text(0.02, 0.95, f"MSE={mse_vx:.4f}", transform=axes[0].transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 绘制y方向速度
        axes[1].plot(steps, cmd_arr[:, 1], 'g-', linewidth=1.5, alpha=0.7, label="Command vy")
        axes[1].plot(steps, vel_arr[:, 1], 'm-', linewidth=1.5, alpha=0.7, label="Actual vy")
        axes[1].set_ylabel("vy (m/s)")
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        axes[1].text(0.02, 0.95, f"MSE={mse_vy:.4f}", transform=axes[1].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 绘制角速度
        axes[2].plot(steps, ang_cmd_arr, 'c-', linewidth=1.5, alpha=0.7, label="Command wz")
        axes[2].plot(steps, ang_vel_arr, 'orange', linewidth=1.5, alpha=0.7, label="Actual wz")
        axes[2].set_ylabel("wz (rad/s)")
        axes[2].set_xlabel("Step")
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
        axes[2].text(0.02, 0.95, f"MSE={mse_wz:.4f}", transform=axes[2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 添加总标题
        fig.suptitle(f"Command vs Actual Velocity (Steps: {min_len})", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        
        # 保存图片
        plot_path = os.path.join(log_dir, "play_cmd_vs_vel.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved command vs velocity plot ({min_len} steps) to {plot_path}")
        
        # 可选：保存数据为CSV文件以便进一步分析
        data_path = os.path.join(log_dir, "cmd_vel_data.csv")
        data = np.column_stack([steps, cmd_arr[:, 0], vel_arr[:, 0], 
                               cmd_arr[:, 1], vel_arr[:, 1], 
                               ang_cmd_arr, ang_vel_arr])
        np.savetxt(data_path, data, delimiter=',',
                   header='step,cmd_vx,act_vx,cmd_vy,act_vy,cmd_wz,act_wz',
                   comments='', fmt='%.6f')
        print(f"[INFO] Saved command/velocity data to {data_path}")
    else:
        print(f"[WARNING] No velocity data recorded. log_cmd={len(log_cmd)}, log_vel={len(log_vel)}")


if __name__ == "__main__":
    EXPORT_POLICY = True
    # run the main execution
    main()
    # close sim app
    simulation_app.close()