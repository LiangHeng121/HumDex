#!/usr/bin/env python3
"""
对齐提示器：对比 Redis 中实时 action_body_*（35D mimic_obs）与参考机器人数据帧。

使用场景：
- xdmocap_teleop.sh 采集的人类操作数据（action_body）与“机器人真实操作”差异很大
- 你想用“机器人数据”里选定的 n 个参考帧作为对齐目标
- 当当前 Redis action_body 与参考帧足够接近（关节差/双臂末端差 < 阈值）时给出提示
- 按键后切换到下一参考帧继续对齐

默认假设：
- Redis key：action_body_unitree_g1_with_hands
- action_body(35D) 结构与本仓库一致：
  [vx, vy, root_z, roll, pitch, yaw_ang_vel, dof_pos(29D)]
  其中 dof_pos 为 unitree_g1 的 29D 关节位置

末端差（可选）：
- 如果环境安装了 mujoco，则使用 XML 做 FK
- 默认 end-effector body 名称使用 g1_mocap_29dof.xml 里的 rubber hand：
  left_rubber_hand / right_rubber_hand
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore


MIMIC_DIM = 35
DOF_DIM = 29


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _find_reference_data_json(ref_path: Path, episode: Optional[str]) -> Path:
    """
    ref_path 可以是：
    - 某个 episode 目录（包含 data.json）
    - 某个 session 目录（包含 episode_*/data.json）
    - 直接就是 data.json 文件
    """
    ref_path = ref_path.resolve()
    if ref_path.is_file() and ref_path.name == "data.json":
        return ref_path

    if ref_path.is_dir():
        direct = ref_path / "data.json"
        if direct.exists():
            return direct

        if episode:
            p = ref_path / episode / "data.json"
            if p.exists():
                return p
            raise FileNotFoundError(f"找不到 episode data.json: {p}")

        # 自动选第一个 episode_*/data.json
        matches = sorted(ref_path.glob("episode_*/data.json"))
        if not matches:
            raise FileNotFoundError(f"在 {ref_path} 下找不到 episode_*/data.json")
        return matches[0]

    raise FileNotFoundError(f"无效 ref_path: {ref_path}")


def _load_frames(data_json: Path) -> List[dict]:
    with data_json.open("r", encoding="utf-8") as f:
        root = json.load(f)
    if not isinstance(root, dict) or "data" not in root or not isinstance(root["data"], list):
        raise ValueError("不支持的 data.json 结构：期望顶层为 dict 且包含 list 类型的 'data' 字段。")
    return root["data"]


def _parse_indices(s: str) -> List[int]:
    """
    支持：
    - "0,10,20"
    - "0:100:10"  (start:stop:step，含 start，不含 stop)
    - "50:200" (step=1)
    """
    ss = str(s).strip()
    if ":" in ss and "," not in ss:
        parts = [p.strip() for p in ss.split(":")]
        if len(parts) == 2:
            start, stop = int(parts[0]), int(parts[1])
            step = 1
        elif len(parts) == 3:
            start, stop, step = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            raise ValueError(f"无法解析 indices: {s}")
        return list(range(start, stop, step))

    out: List[int] = []
    for p in [x.strip() for x in ss.split(",") if x.strip()]:
        out.append(int(p))
    return out


def _mimic_from_frame(fr: dict) -> np.ndarray:
    a = fr.get("action_body", None)
    if a is None:
        raise KeyError("参考帧缺少 action_body")
    arr = np.asarray(a, dtype=np.float32).reshape(-1)
    if arr.size != MIMIC_DIM:
        raise ValueError(f"action_body 维度不匹配：期望 {MIMIC_DIM}，实际 {arr.size}")
    return arr


def _mimic_from_redis_bytes(b: Optional[bytes]) -> Optional[np.ndarray]:
    if b is None:
        return None
    try:
        x = json.loads(b)
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if arr.size != MIMIC_DIM:
            return None
        return arr
    except Exception:
        return None


def _quat_wxyz_from_roll_pitch(roll: float, pitch: float) -> np.ndarray:
    """
    用 roll/pitch 构造四元数（yaw=0）。返回 wxyz。
    这里用欧拉角合成：R = Rz(0)*Ry(pitch)*Rx(roll) = Ry(pitch)*Rx(roll)
    """
    cr = float(np.cos(roll * 0.5))
    sr = float(np.sin(roll * 0.5))
    cp = float(np.cos(pitch * 0.5))
    sp = float(np.sin(pitch * 0.5))
    # yaw = 0 => cy=1, sy=0
    # quat (wxyz) for yaw-pitch-roll (Z-Y-X):
    # w = cy*cp*cr + sy*sp*sr = cp*cr
    # x = cy*cp*sr - sy*sp*cr = cp*sr
    # y = sy*cp*sr + cy*sp*cr = sp*cr
    # z = sy*cp*cr - cy*sp*sr = -sp*sr
    w = cp * cr
    x = cp * sr
    y = sp * cr
    z = -sp * sr
    q = np.asarray([w, x, y, z], dtype=np.float32)
    n = float(np.linalg.norm(q) + 1e-12)
    return q / n


@dataclass
class FKContext:
    mj: object
    model: object
    data: object
    left_body_id: int
    right_body_id: int


def _try_init_fk(xml_path: Path, left_body: str, right_body: str) -> Optional[FKContext]:
    """
    尝试用 mujoco 做 FK。失败则返回 None（脚本仍可跑 joint diff）。
    """
    try:
        import mujoco as mj  # type: ignore
    except Exception:
        return None

    try:
        model = mj.MjModel.from_xml_path(str(xml_path))
        data = mj.MjData(model)
        left_id = model.body(left_body).id
        right_id = model.body(right_body).id
        return FKContext(mj=mj, model=model, data=data, left_body_id=left_id, right_body_id=right_id)
    except Exception:
        return None


def _eef_positions_from_mimic(mimic35: np.ndarray, fk: FKContext) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (left_eef_xyz, right_eef_xyz)，单位为 MuJoCo world 坐标系单位（通常米）。
    """
    m = np.asarray(mimic35, dtype=np.float32).reshape(MIMIC_DIM)
    root_z = float(m[2])
    roll = float(m[3])
    pitch = float(m[4])
    dof = m[6:].astype(np.float32).reshape(DOF_DIM)

    qpos = fk.data.qpos.copy()
    # freejoint: pos(3) + quat(4)
    qpos[0:3] = np.asarray([0.0, 0.0, root_z], dtype=np.float32)
    qpos[3:7] = _quat_wxyz_from_roll_pitch(roll, pitch)
    qpos[7 : 7 + DOF_DIM] = dof
    fk.data.qpos[:] = qpos
    fk.mj.mj_forward(fk.model, fk.data)
    left = np.asarray(fk.data.xpos[fk.left_body_id], dtype=np.float32).reshape(3)
    right = np.asarray(fk.data.xpos[fk.right_body_id], dtype=np.float32).reshape(3)
    return left, right


class _KeyPoller:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def poll(self) -> Optional[str]:
        dr, _, _ = select.select([sys.stdin], [], [], 0.0)
        if dr:
            ch = sys.stdin.read(1)
            return ch
        return None


def _fmt_vec(v: np.ndarray) -> str:
    v = np.asarray(v).reshape(-1)
    return "[" + ", ".join(f"{float(x):+.3f}" for x in v.tolist()) + "]"


def main() -> int:
    p = argparse.ArgumentParser(description="对比 Redis action_body 与参考帧，满足阈值时提示，并按键推进参考帧。")
    p.add_argument("--ref_dir", type=str, required=True, help="参考机器人数据目录（session/episode/data.json）")
    p.add_argument("--ref_episode", type=str, default="", help="当 ref_dir 为 session 目录时，指定 episode_XXXX")
    p.add_argument("--ref_indices", type=str, required=True, help='参考帧 idx 列表，如 "0,10,20" 或 "0:300:10"')
    p.add_argument("--redis_ip", type=str, default="localhost")
    p.add_argument("--robot_key", type=str, default="unitree_g1_with_hands")
    p.add_argument("--redis_key_action_body", type=str, default="", help="覆盖 action_body redis key（默认 action_body_{robot_key}）")

    p.add_argument("--joint_linf_thr", type=float, default=0.10, help="关节（29D dof_pos）最大绝对误差阈值（rad）")
    p.add_argument("--joint_rms_thr", type=float, default=0.06, help="关节（29D dof_pos）RMS 误差阈值（rad）")
    p.add_argument("--eef_thr_m", type=float, default=0.06, help="左右手末端位置误差阈值（m），需要 mujoco FK")

    p.add_argument("--fk", action="store_true", help="启用 mujoco FK 计算末端差（若环境无 mujoco 会自动降级）")
    p.add_argument("--fk_xml", type=str, default=str(_repo_root() / "GMR" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"))
    p.add_argument("--fk_left_body", type=str, default="left_rubber_hand")
    p.add_argument("--fk_right_body", type=str, default="right_rubber_hand")

    p.add_argument("--rate_hz", type=float, default=30.0)
    p.add_argument("--next_key", type=str, default="n", help="满足阈值后，按该键切换下一参考帧")
    p.add_argument("--print_key", type=str, default="p", help="按该键打印一次当前详细误差")
    p.add_argument("--quit_key", type=str, default="q")
    args = p.parse_args()

    ref_dir = Path(args.ref_dir)
    ref_episode = str(args.ref_episode).strip() or None
    ref_data_json = _find_reference_data_json(ref_dir, ref_episode)
    frames = _load_frames(ref_data_json)

    ref_indices = _parse_indices(args.ref_indices)
    if not ref_indices:
        raise ValueError("ref_indices 为空")

    # build reference list (by idx field)
    idx_to_frame: Dict[int, dict] = {}
    for fr in frames:
        if "idx" in fr:
            try:
                idx_to_frame[int(fr["idx"])] = fr
            except Exception:
                pass

    refs: List[Tuple[int, np.ndarray]] = []
    for i in ref_indices:
        if i not in idx_to_frame:
            raise KeyError(f"参考 idx={i} 不存在于 {ref_data_json}（可用 idx 数：{len(idx_to_frame)}）")
        refs.append((i, _mimic_from_frame(idx_to_frame[i])))

    # redis
    try:
        import redis  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少 redis python 包，请在 conda env gmr 中运行该脚本。") from e

    robot_key = str(args.robot_key).strip()
    key_action_body = str(args.redis_key_action_body).strip() or f"action_body_{robot_key}"
    client = redis.Redis(host=str(args.redis_ip), port=6379, db=0, decode_responses=False)
    try:
        client.ping()
    except Exception as e:
        raise RuntimeError(f"Redis 连接失败: {e}")

    fk_ctx: Optional[FKContext] = None
    if bool(args.fk):
        fk_ctx = _try_init_fk(Path(args.fk_xml), str(args.fk_left_body), str(args.fk_right_body))
        if fk_ctx is None:
            print("❌  你开启了 --fk，但 mujoco FK 初始化失败/未安装 mujoco。")
            print("    这会导致无法计算“手臂末端差别”，请安装 mujoco 后重试，或去掉 --fk 只做关节对比。")
            return 2

    print("")
    print("=== ActionBody 对齐提示器 ===")
    print(f"- ref_data_json      : {ref_data_json}")
    print(f"- ref_indices        : {ref_indices}")
    print(f"- redis_ip           : {args.redis_ip}")
    print(f"- redis_key_action_body: {key_action_body}")
    if fk_ctx is not None:
        print(
            f"- thresholds         : joint_linf<{args.joint_linf_thr:.3f} rad, joint_rms<{args.joint_rms_thr:.3f} rad, eef<{args.eef_thr_m:.3f} m"
        )
    else:
        print(f"- thresholds         : joint_linf<{args.joint_linf_thr:.3f} rad, joint_rms<{args.joint_rms_thr:.3f} rad")
    print(f"- keys               : next='{args.next_key}', print='{args.print_key}', quit='{args.quit_key}'")
    if fk_ctx is not None:
        print(f"- fk                 : enabled (xml={args.fk_xml}, left_body={args.fk_left_body}, right_body={args.fk_right_body})")
    else:
        print("- fk                 : disabled/unavailable")
    print("")

    sleep_dt = 1.0 / max(1e-6, float(args.rate_hz))

    cur = 0
    last_close = False

    def _compute_metrics(cur_m: np.ndarray, ref_m: np.ndarray) -> Dict[str, float]:
        cur_dof = cur_m[6:].reshape(DOF_DIM)
        ref_dof = ref_m[6:].reshape(DOF_DIM)
        diff = (cur_dof - ref_dof).astype(np.float32)
        linf = float(np.max(np.abs(diff)))
        rms = float(np.sqrt(np.mean(diff * diff)))
        out = {"joint_linf": linf, "joint_rms": rms}
        if fk_ctx is not None:
            try:
                le, re = _eef_positions_from_mimic(cur_m, fk_ctx)
                le_r, re_r = _eef_positions_from_mimic(ref_m, fk_ctx)
                out["eef_left"] = float(np.linalg.norm(le - le_r))
                out["eef_right"] = float(np.linalg.norm(re - re_r))
                out["eef_max"] = max(out["eef_left"], out["eef_right"])
            except Exception:
                pass
        return out

    with _KeyPoller() as kp:
        while True:
            key = kp.poll()
            if key is not None:
                if key == str(args.quit_key):
                    print("\n退出。")
                    return 0
                if key == str(args.print_key):
                    # 强制打印一次
                    cur_raw = client.get(key_action_body)
                    cur_m = _mimic_from_redis_bytes(cur_raw)
                    if cur_m is None:
                        print("（Redis action_body 缺失/格式错误）")
                    else:
                        ref_idx, ref_m = refs[cur]
                        m = _compute_metrics(cur_m, ref_m)
                        print(f"[ref idx={ref_idx}] joint_linf={m.get('joint_linf', -1):.4f}, joint_rms={m.get('joint_rms', -1):.4f}, eef_max={m.get('eef_max', -1):.4f}")
                    sys.stdout.flush()
                if key == str(args.next_key):
                    cur = min(cur + 1, len(refs) - 1)
                    last_close = False
                    ref_idx, _ = refs[cur]
                    print(f"\n➡️  切换到下一参考帧：[{cur+1}/{len(refs)}] idx={ref_idx}")
                    sys.stdout.flush()

            # read redis
            cur_raw = client.get(key_action_body)
            cur_m = _mimic_from_redis_bytes(cur_raw)
            if cur_m is None:
                # 避免刷屏
                sys.stdout.write("\r等待 Redis action_body...".ljust(80))
                sys.stdout.flush()
                import time

                time.sleep(sleep_dt)
                continue

            ref_idx, ref_m = refs[cur]
            metrics = _compute_metrics(cur_m, ref_m)

            joint_ok = (metrics["joint_linf"] <= float(args.joint_linf_thr)) and (metrics["joint_rms"] <= float(args.joint_rms_thr))
            eef_ok = True
            if fk_ctx is not None and "eef_max" in metrics:
                eef_ok = metrics["eef_max"] <= float(args.eef_thr_m)

            close = bool(joint_ok and eef_ok)

            # 单行状态
            eef_max = metrics.get("eef_max", float("nan"))
            msg = f"[{cur+1}/{len(refs)} idx={ref_idx}] joint_linf={metrics['joint_linf']:.4f} joint_rms={metrics['joint_rms']:.4f}"
            if fk_ctx is not None and "eef_max" in metrics:
                msg += f" eef_max={eef_max:.4f}m"
            msg += "  "
            sys.stdout.write("\r" + msg.ljust(110))
            sys.stdout.flush()

            if close and not last_close:
                print("")
                if fk_ctx is not None and "eef_max" in metrics:
                    print(
                        f"✅ 已接近参考帧 idx={ref_idx}：joint_linf={metrics['joint_linf']:.4f}, joint_rms={metrics['joint_rms']:.4f}, eef_max={metrics['eef_max']:.4f}m"
                    )
                else:
                    print(f"✅ 已接近参考帧 idx={ref_idx}：joint_linf={metrics['joint_linf']:.4f}, joint_rms={metrics['joint_rms']:.4f}")
                print(f"按 '{args.next_key}' 进入下一参考帧；按 '{args.print_key}' 打印详情；按 '{args.quit_key}' 退出。")
                sys.stdout.flush()

            last_close = close

            import time

            time.sleep(sleep_dt)


if __name__ == "__main__":
    raise SystemExit(main())


