#!/usr/bin/env python3
"""
离线可视化：录制数据里的 action_wuji_qpos_target_{left/right} (20D)。

实现参考：deploy_real/policy_inference.py 里的 eval/replay 可视化逻辑（复用 HandVisualizer）。

用法示例：
  - 指定 task_dir（自动找最新 episode）：
      python deploy_real/viz_recorded_wuji_qpos_target.py /path/to/deploy_real/twist2_demonstration/20260115_0905 --hand_side right
  - 指定 episode_dir：
      python deploy_real/viz_recorded_wuji_qpos_target.py /path/to/.../episode_0000 --hand_side right
  - 指定 data.json：
      python deploy_real/viz_recorded_wuji_qpos_target.py /path/to/.../episode_0000/data.json --hand_side right
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _resolve_data_json(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if p.is_file() and p.name.endswith(".json"):
        return p
    if p.is_dir():
        episodes = sorted([x for x in p.iterdir() if x.is_dir() and x.name.startswith("episode_")])
        if episodes:
            return (episodes[-1] / "data.json").resolve()
        candidate = p / "data.json"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"无法定位 data.json: {p}")


def _load_episode_items(data_json: Path) -> List[Dict[str, Any]]:
    with open(data_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    items = obj.get("data", None)
    if not isinstance(items, list):
        raise ValueError(f"data.json 格式不对：缺少 data(list)，path={data_json}")
    return items


def _extract_action_20(item: Dict[str, Any], *, hand_side: str) -> Optional[np.ndarray]:
    s = str(hand_side).lower()
    assert s in ["left", "right"]
    key = f"action_wuji_qpos_target_{s}"
    v = item.get(key, None)
    if v is None:
        return None
    try:
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.shape[0] != 20:
        return None
    return arr


def _extract_t_action_wuji(item: Dict[str, Any], *, hand_side: str) -> Optional[int]:
    s = str(hand_side).lower()
    assert s in ["left", "right"]
    key = f"t_action_wuji_hand_{s}"
    v = item.get(key, None)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _viz_matplotlib(actions: np.ndarray, ts_ms: Optional[np.ndarray], fps: float, stream: bool, save: str) -> int:
    """
    Fallback visualizer when mujoco is not available:
      - 5 subplots (thumb/index/middle/ring/little)
      - each subplot shows 4 joint values over time (lines)
      - a moving vertical cursor indicates current frame
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少 matplotlib，无法 fallback 可视化。") from e

    actions = np.asarray(actions, dtype=np.float32).reshape(-1, 20)
    T = int(actions.shape[0])
    t = np.arange(T, dtype=np.float32) / max(1e-6, float(fps))

    names = ["Thumb", "Index", "Middle", "Ring", "Little"]
    fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("Recorded action_wuji_qpos_target (20D) - fallback (no mujoco)")

    # plot lines
    lines = []
    cursors = []
    for fi in range(5):
        ax = axs[fi]
        base = fi * 4
        for j in range(4):
            (ln,) = ax.plot(t, actions[:, base + j], lw=1.5, label=f"j{j}")
            lines.append(ln)
        cur = ax.axvline(0.0, color="k", lw=1.5)
        cursors.append(cur)
        ax.set_ylabel(names[fi])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=4, fontsize=8)

    axs[-1].set_xlabel("time (s)")
    txt = fig.text(0.01, 0.01, "", fontsize=10)

    # Optional save via matplotlib animation (requires ffmpeg for mp4)
    if save:
        try:
            from matplotlib.animation import FFMpegWriter  # type: ignore
        except Exception:
            FFMpegWriter = None  # type: ignore

    if not stream and not save:
        print("[fallback] no_stream 且未设置 --save：将只打印统计，不弹窗。")
        print(f"[fallback] actions shape={actions.shape}, duration~{T/max(1e-6,fps):.1f}s")
        return 0

    plt.ion()
    if stream:
        plt.show()

    writer = None
    if save:
        save_path = str(Path(save).expanduser().resolve())
        if save_path.lower().endswith(".mp4"):
            if "FFMpegWriter" in globals() and FFMpegWriter is not None:
                writer = FFMpegWriter(fps=int(round(float(fps))))
            else:
                print("[fallback] 无法保存 mp4：缺少 matplotlib 的 ffmpeg writer。可先安装 ffmpeg 或改用窗口预览。")
                writer = None
        else:
            print("[fallback] --save 目前推荐 .mp4（需要 ffmpeg）。")
            writer = None

    if writer is not None:
        with writer.saving(fig, save_path, dpi=120):
            for i in range(T):
                x = float(t[i])
                for cur in cursors:
                    cur.set_xdata([x, x])
                if ts_ms is not None:
                    txt.set_text(f"frame={i}/{T-1}  t={x:.3f}s  t_action_wuji_ms={int(ts_ms[i])}")
                else:
                    txt.set_text(f"frame={i}/{T-1}  t={x:.3f}s")
                fig.canvas.draw_idle()
                writer.grab_frame()
        print(f"[fallback] saved: {save_path}")
        return 0

    # interactive playback
    dt = 1.0 / max(1e-6, float(fps))
    for i in range(T):
        x = float(t[i])
        for cur in cursors:
            cur.set_xdata([x, x])
        if ts_ms is not None:
            txt.set_text(f"frame={i}/{T-1}  t={x:.3f}s  t_action_wuji_ms={int(ts_ms[i])}")
        else:
            txt.set_text(f"frame={i}/{T-1}  t={x:.3f}s")
        fig.canvas.draw_idle()
        plt.pause(0.001)
        time.sleep(dt)

    plt.ioff()
    plt.show()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="离线可视化 episode 中的 action_wuji_qpos_target (20D)")
    ap.add_argument("path", type=str, help="task_dir / episode_dir / data.json 路径")
    ap.add_argument("--hand_side", type=str, default="right", choices=["left", "right"], help="选择可视化哪只手")
    ap.add_argument("--fps", type=float, default=30.0, help="播放帧率（窗口预览/保存视频）")
    ap.add_argument("--start", type=int, default=0, help="起始帧 index")
    ap.add_argument("--max_frames", type=int, default=-1, help="最多播放帧数（-1=全部）")
    ap.add_argument("--warmup_steps", type=int, default=50, help="预热步数（把手从 neutral 推到第一帧动作）")
    ap.add_argument("--no_stream", action="store_true", help="不弹窗预览（适合无显示器环境；可配合 --save）")
    ap.add_argument("--save", type=str, default="", help="保存 mp4 路径（需要 imageio/imageio-ffmpeg）")
    args = ap.parse_args()

    data_json = _resolve_data_json(args.path)
    items = _load_episode_items(data_json)

    # Resolve sim_viz imports like policy_inference.py
    repo_root = Path(__file__).resolve().parents[1]
    sim_viz_dir = repo_root / "act" / "sim_viz"
    if str(sim_viz_dir) not in os.sys.path:
        os.sys.path.insert(0, str(sim_viz_dir))
    HandVisualizer = None
    get_default_paths = None
    save_video = None
    _viz_import_error = None
    try:
        from visualizers import HandVisualizer as _HV, get_default_paths as _GDP, save_video as _SV  # type: ignore

        HandVisualizer = _HV
        get_default_paths = _GDP
        save_video = _SV
    except Exception as e:
        _viz_import_error = e

    # Headless safety
    stream = not bool(args.no_stream)
    if stream and (os.environ.get("DISPLAY") is None and os.environ.get("WAYLAND_DISPLAY") is None):
        print("[GUI] 未检测到 DISPLAY/WAYLAND_DISPLAY，自动关闭窗口预览（仍可用 --save 输出 mp4）。")
        stream = False

    # Collect actions first (so we can fallback cleanly)
    start = max(0, int(args.start))
    end = len(items) if int(args.max_frames) <= 0 else min(len(items), start + int(args.max_frames))
    actions_list = []
    ts_list = []
    for i in range(start, end):
        a = _extract_action_20(items[i], hand_side=str(args.hand_side))
        if a is None:
            continue
        actions_list.append(a)
        ts_list.append(_extract_t_action_wuji(items[i], hand_side=str(args.hand_side)))
    if not actions_list:
        raise RuntimeError(f"在区间 [{start},{end}) 内没找到有效的 action_wuji_qpos_target_{args.hand_side}")
    actions = np.stack(actions_list, axis=0)  # (T,20)
    ts_ms = None
    if any(x is not None for x in ts_list):
        ts_ms = np.asarray([(-1 if x is None else int(x)) for x in ts_list], dtype=np.int64)

    # If mujoco is unavailable, fallback to matplotlib plot.
    if HandVisualizer is None or get_default_paths is None:
        print(f"[WARN] mujoco/HandVisualizer 不可用，将使用 matplotlib fallback 可视化 20D 数值曲线。error={_viz_import_error}")
        return _viz_matplotlib(actions, ts_ms, fps=float(args.fps), stream=stream, save=str(args.save))

    # Setup HandVisualizer (MuJoCo render)
    paths = get_default_paths()
    xml_key = f"{args.hand_side}_hand_xml"
    if xml_key not in paths:
        raise ValueError(f"找不到 {xml_key}，paths={list(paths.keys())}")
    viz = HandVisualizer(paths[xml_key], hand_side=str(args.hand_side))

    # Optional window
    writer_frames: List[np.ndarray] = []
    if stream:
        try:
            import cv2  # type: ignore

            cv2.namedWindow("Wuji Hand (20D) - Preview", cv2.WINDOW_NORMAL)
        except Exception as e:
            print(f"[GUI] 打开窗口失败，改为 no_stream：{e}")
            stream = False

    # warmup
    first_action = np.asarray(actions[0], dtype=np.float32).reshape(20)
    for _ in range(max(0, int(args.warmup_steps))):
        viz.step(first_action)

    dt = 1.0 / max(1e-3, float(args.fps))
    shown = 0
    for a in actions:
        frame = viz.step(np.asarray(a, dtype=np.float32))
        shown += 1

        if args.save:
            writer_frames.append(frame)

        if stream:
            import cv2  # type: ignore

            cv2.imshow("Wuji Hand (20D) - Preview", frame[:, :, ::-1])  # RGB->BGR
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        time.sleep(dt)

    if stream:
        try:
            import cv2  # type: ignore

            cv2.destroyAllWindows()
        except Exception:
            pass

    if args.save:
        save_path = str(Path(args.save).expanduser().resolve())
        ok = bool(save_video(writer_frames, save_path, fps=int(round(float(args.fps))))) if save_video is not None else False
        if not ok:
            return 2

    print(f"[done] frames_rendered={shown}, data_json={data_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


