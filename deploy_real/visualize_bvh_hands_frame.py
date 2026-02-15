#!/usr/bin/env python3
"""
Visualize BVH hands (3D joint points) for a given frame.

默认：取第 0 帧（第一帧），把 BVH 里的左右手关节（RightHand/RightThumbFinger...）做 3D 散点图。

可选：
- overlay tracking26：把 replay_bvh_wuji_to_redis.py 的 26D hand_tracking 映射点也叠加画出来（用于检查映射大概位置）
- 保存 png（无 GUI 环境也能用）

依赖：
- matplotlib（建议在 gmr 环境里跑）
"""

import argparse
import os
import sys
import subprocess
import tempfile
from typing import Dict, Any, List, Tuple, Optional

import numpy as np


def _resolve_bvh_path(bvh_file: str) -> str:
    bvh_path = bvh_file
    if not os.path.isabs(bvh_path) and not os.path.exists(bvh_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cand = os.path.join(repo_root, bvh_path)
        if os.path.exists(cand):
            bvh_path = cand
    return bvh_path


def _load_frames(bvh_path: str, fmt: str) -> Tuple[List[Dict[str, Any]], float]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from deploy_real.replay_bvh_body_to_redis import _load_bvh_frames_via_gmr  # type: ignore

    return _load_bvh_frames_via_gmr(bvh_path, fmt)


def _get_pos(frame: Dict[str, Any], name: str) -> np.ndarray:
    v = frame.get(name, None)
    if isinstance(v, (list, tuple)) and len(v) >= 1:
        return np.asarray(v[0], dtype=np.float32).reshape(3)
    raise KeyError(name)


def _get_quat_wxyz(frame: Dict[str, Any], name: str) -> np.ndarray:
    v = frame.get(name, None)
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        q = np.asarray(v[1], dtype=np.float32).reshape(4)
        return q
    raise KeyError(name)


def _parse_bvh_hierarchy_parents_and_endsites(bvh_path: str) -> Tuple[Dict[str, Optional[str]], Dict[str, np.ndarray]]:
    """
    只解析 BVH 的 HIERARCHY 部分（到 MOTION 之前），提取：
    - parent[name] = parent_name
    - endsite_offset[name] = np.array([x,y,z])  # End Site 的 OFFSET（单位：cm，BVH 坐标系）

    注意：End Site 自身在 BVH 里没有名字，我们把它挂到“所属 JOINT”上。
    """
    parent: Dict[str, Optional[str]] = {}
    endsite_offset: Dict[str, np.ndarray] = {}

    joint_stack: List[str] = []
    in_end_site = False
    end_owner: Optional[str] = None

    with open(bvh_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.upper().startswith("MOTION"):
                break

            if line.startswith("ROOT "):
                name = line.split()[1]
                parent[name] = None
                joint_stack.append(name)
                continue

            if line.startswith("JOINT "):
                name = line.split()[1]
                pj = joint_stack[-1] if joint_stack else None
                parent[name] = pj
                joint_stack.append(name)
                continue

            if line.startswith("End Site"):
                in_end_site = True
                end_owner = joint_stack[-1] if joint_stack else None
                continue

            if in_end_site and line.startswith("OFFSET "):
                parts = line.split()
                if len(parts) == 4 and end_owner:
                    endsite_offset[end_owner] = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
                continue

            if line.startswith("}"):
                if in_end_site:
                    # closing End Site block
                    in_end_site = False
                    end_owner = None
                    continue
                # closing a JOINT/ROOT block
                if joint_stack:
                    joint_stack.pop()
                continue

    return parent, endsite_offset


def _rotate_vec_by_quat_wxyz(vec: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Rotate 3D vector by quaternion in wxyz order.
    """
    try:
        from scipy.spatial.transform import Rotation as R  # type: ignore
    except Exception as e:
        raise RuntimeError(f"需要 scipy 才能计算 End Site 位置：{e}") from e

    # scipy uses xyzw (scalar-last)
    q = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
    return R.from_quat(q).apply(vec)


def _compute_end_site_pos_for_joint(
    frame: Dict[str, Any],
    joint: str,
    endsite_offsets_cm: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    """
    计算某个 joint 的 End Site 世界坐标（在 loader 的坐标系下），若该 joint 没有 End Site OFFSET 则返回 None。
    """
    if joint not in endsite_offsets_cm:
        return None
    if joint not in frame:
        return None
    off_cm = endsite_offsets_cm[joint]
    try:
        jpos = _get_pos(frame, joint)
        jq = _get_quat_wxyz(frame, joint)
    except Exception:
        return None

    # loader coordinate transform (must match replay loader)
    rot_m = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    off_m = (off_cm / 100.0) @ rot_m.T
    delta = _rotate_vec_by_quat_wxyz(off_m, jq)
    return jpos + delta.astype(np.float32)


def _compute_wrist_to_fingertip_distances_m(
    frame: Dict[str, Any],
    side: str,
    endsite_offsets_cm: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    计算“手腕(Hand) -> 五指指尖”的距离（米）。
    指尖优先用 End Site（真正末端），缺失则 fallback 到最后一节关节位置。
    """
    side = side.lower()
    assert side in ["left", "right"]
    pfx = "Left" if side == "left" else "Right"

    wrist_joint = f"{pfx}Hand"
    wrist = _get_pos(frame, wrist_joint)

    # BVH 中每根手指的最后一节关节名（你的 BVH 命名）
    tip_joints = {
        "thumb": f"{pfx}ThumbFinger2",
        "index": f"{pfx}IndexFinger3",
        "middle": f"{pfx}MiddleFinger3",
        "ring": f"{pfx}RingFinger3",
        "pinky": f"{pfx}PinkyFinger3",
    }

    out: Dict[str, float] = {}
    for finger, j in tip_joints.items():
        tip = _compute_end_site_pos_for_joint(frame, j, endsite_offsets_cm)
        if tip is None:
            # fallback: use the last joint position
            tip = _get_pos(frame, j)
        out[finger] = float(np.linalg.norm(tip - wrist))
    return out


def _collect_bvh_hand_points(frame: Dict[str, Any], side: str) -> Dict[str, np.ndarray]:
    """
    收集 BVH 手部关节的世界坐标点（只画 JOINT，不含 End Site）。
    side: 'left'/'right'
    """
    side = side.lower()
    assert side in ["left", "right"]
    pfx = "Left" if side == "left" else "Right"

    # 你的 BVH 里手骨骼命名（参考 xdmocap 的 list / BVH hierarchy）
    names = [
        f"{pfx}Hand",
        f"{pfx}ThumbFinger",
        f"{pfx}ThumbFinger1",
        f"{pfx}ThumbFinger2",
        f"{pfx}IndexFinger",
        f"{pfx}IndexFinger1",
        f"{pfx}IndexFinger2",
        f"{pfx}IndexFinger3",
        f"{pfx}MiddleFinger",
        f"{pfx}MiddleFinger1",
        f"{pfx}MiddleFinger2",
        f"{pfx}MiddleFinger3",
        f"{pfx}RingFinger",
        f"{pfx}RingFinger1",
        f"{pfx}RingFinger2",
        f"{pfx}RingFinger3",
        f"{pfx}PinkyFinger",
        f"{pfx}PinkyFinger1",
        f"{pfx}PinkyFinger2",
        f"{pfx}PinkyFinger3",
    ]

    pts: Dict[str, np.ndarray] = {}
    for n in names:
        if n in frame:
            pts[n] = _get_pos(frame, n)
    return pts


def _collect_tracking26_points(frame: Dict[str, Any], side: str) -> Dict[str, np.ndarray]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from deploy_real.replay_bvh_wuji_to_redis import _bvh_hand_to_tracking26  # type: ignore

    d = _bvh_hand_to_tracking26(frame, side)
    pts: Dict[str, np.ndarray] = {}
    for k, v in d.items():
        # v = [[x,y,z], [qw,qx,qy,qz]]
        try:
            pts[k] = np.asarray(v[0], dtype=np.float32).reshape(3)
        except Exception:
            continue
    return pts


def _set_axes_equal(ax):
    # Make 3D plot have equal aspect ratio
    xs, ys, zs = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    xmid = 0.5 * (xs[0] + xs[1])
    ymid = 0.5 * (ys[0] + ys[1])
    zmid = 0.5 * (zs[0] + zs[1])
    r = 0.5 * max(xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0])
    ax.set_xlim3d(xmid - r, xmid + r)
    ax.set_ylim3d(ymid - r, ymid + r)
    ax.set_zlim3d(zmid - r, zmid + r)


def _compute_bounds(points_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    P = np.concatenate([p.reshape(-1, 3) for p in points_list if p is not None and p.size > 0], axis=0)
    mn = np.min(P, axis=0)
    mx = np.max(P, axis=0)
    return mn, mx


def _apply_bounds_equal(ax, mn: np.ndarray, mx: np.ndarray):
    xmid, ymid, zmid = 0.5 * (mn + mx)
    r = 0.55 * float(np.max(mx - mn))  # a bit padded
    ax.set_xlim3d(xmid - r, xmid + r)
    ax.set_ylim3d(ymid - r, ymid + r)
    ax.set_zlim3d(zmid - r, zmid + r)

def _apply_fixed_range(ax, r: float):
    ax.set_xlim3d(-r, r)
    ax.set_ylim3d(-r, r)
    ax.set_zlim3d(-r, r)

def _transform_points(
    pts: Dict[str, np.ndarray],
    end_pts: Dict[str, np.ndarray],
    origin: Optional[np.ndarray],
    scale: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Translate by origin (if provided) and scale points.
    """
    def tf(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for k, p in d.items():
            q = p.astype(np.float32)
            if origin is not None:
                q = q - origin
            q = q * float(scale)
            out[k] = q
        return out
    return tf(pts), tf(end_pts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize BVH hand 3D points for a given frame")
    ap.add_argument("--bvh_file", type=str, required=True)
    ap.add_argument("--format", choices=["lafan1", "nokov"], default="nokov")
    ap.add_argument("--frame", type=int, default=0, help="Frame index (default: 0)")
    ap.add_argument("--overlay_tracking26", action="store_true", help="Overlay Wuji tracking26 mapped points")
    ap.add_argument("--labels", action="store_true", help="Draw text labels (may be cluttered)")
    ap.add_argument("--lines", action="store_true", help="Draw skeleton lines based on BVH hierarchy (hands only)")
    ap.add_argument("--end_sites", action="store_true", help="Include End Site points (finger tips) if present in BVH")
    ap.add_argument("--print_wrist_to_tips", action="store_true", help="Print wrist->5 fingertips distances for frame 0")
    ap.add_argument("--video_out", type=str, default="", help="Export an mp4 video by rendering frames (requires ffmpeg)")
    ap.add_argument("--video_fps", type=float, default=30.0, help="Video FPS (default: 30)")
    ap.add_argument("--video_start", type=int, default=0, help="Video start frame index (inclusive)")
    ap.add_argument("--video_end", type=int, default=300, help="Video end frame index (exclusive)")
    ap.add_argument("--video_layout", choices=["both", "left", "right"], default="both", help="Video layout: both hands or a single hand full-frame")
    ap.add_argument("--video_zoom", choices=["global", "per_hand"], default="per_hand", help="Axis zoom mode for video")
    ap.add_argument("--video_coords", choices=["world", "wrist_local"], default="wrist_local", help="Coordinate mode for video")
    ap.add_argument("--video_scale", type=float, default=100.0, help="Scale factor for visualization (default: 100 => meters->cm)")
    ap.add_argument("--video_fixed_range", type=float, default=0.0, help="If >0, use fixed axis range [-r,r] (in scaled units) for video")
    ap.add_argument("--video_dpi", type=int, default=180, help="DPI for rendered frames (default: 180)")
    ap.add_argument("--video_point_size", type=int, default=28, help="Joint point size for video (default: 28)")
    ap.add_argument("--video_end_point_size", type=int, default=26, help="End site point size for video (default: 26)")
    ap.add_argument("--video_line_width", type=float, default=2.0, help="Skeleton line width for video (default: 2.0)")
    ap.add_argument("--out", type=str, default="", help="Save figure to file (png). If empty, do not save.")
    ap.add_argument("--no_show", action="store_true", help="Do not open a window (use with --out on headless).")
    args = ap.parse_args()

    bvh_path = _resolve_bvh_path(args.bvh_file)
    if not os.path.exists(bvh_path):
        print(f"❌ 找不到 BVH 文件: {args.bvh_file}", file=sys.stderr)
        return 2

    frames, _hh = _load_frames(bvh_path, args.format)
    if not frames:
        print("❌ BVH 没有帧数据", file=sys.stderr)
        return 2

    # Parse hierarchy once if needed (lines/end_sites/distances/video)
    parent_map: Dict[str, Optional[str]] = {}
    endsite_offsets_cm: Dict[str, np.ndarray] = {}
    if args.lines or args.end_sites or args.print_wrist_to_tips or args.video_out:
        parent_map, endsite_offsets_cm = _parse_bvh_hierarchy_parents_and_endsites(bvh_path)

    # Print distances for frame 0 (requested)
    if args.print_wrist_to_tips:
        f0 = frames[0]
        try:
            dl = _compute_wrist_to_fingertip_distances_m(f0, "left", endsite_offsets_cm)
            dr = _compute_wrist_to_fingertip_distances_m(f0, "right", endsite_offsets_cm)
            print("[frame0] wrist->fingertip distances (m)")
            print(f"  left : thumb={dl['thumb']:.4f}, index={dl['index']:.4f}, middle={dl['middle']:.4f}, ring={dl['ring']:.4f}, pinky={dl['pinky']:.4f}")
            print(f"  right: thumb={dr['thumb']:.4f}, index={dr['index']:.4f}, middle={dr['middle']:.4f}, ring={dr['ring']:.4f}, pinky={dr['pinky']:.4f}")
        except Exception as e:
            print(f"❌ 计算 wrist->tips 距离失败：{e}", file=sys.stderr)

    # If exporting video, render a range and compose with ffmpeg
    if args.video_out:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as e:
            print(f"❌ 缺少 matplotlib，无法导出视频：{e}", file=sys.stderr)
            return 3

        vstart = max(0, int(args.video_start))
        vend = min(len(frames), int(args.video_end))
        if vstart >= vend:
            print(f"❌ video_start/end 非法: start={vstart}, end={vend}, total={len(frames)}", file=sys.stderr)
            return 2

        # Precompute bounds to avoid axis jitter (in the chosen coordinate mode).
        # For readability, default is per-hand bounds (each subplot zooms to its own hand).
        mn_glb = mx_glb = None
        mn_l = mx_l = None
        mn_r = mx_r = None
        pts_glb: List[np.ndarray] = []
        pts_l: List[np.ndarray] = []
        pts_r: List[np.ndarray] = []

        def add_bounds_points(side: str, fr: Dict[str, Any], store: List[np.ndarray]):
            pts = _collect_bvh_hand_points(fr, side)
            if pts:
                store.append(np.stack(list(pts.values()), axis=0))
            if args.end_sites and endsite_offsets_cm:
                pfx = "Left" if side == "left" else "Right"
                for joint in endsite_offsets_cm.keys():
                    if not joint.startswith(pfx):
                        continue
                    if not (("Hand" in joint) or ("Finger" in joint)):
                        continue
                    p = _compute_end_site_pos_for_joint(fr, joint, endsite_offsets_cm)
                    if p is not None:
                        store.append(p.reshape(1, 3))

        for i in range(vstart, vend):
            fr = frames[i]
            add_bounds_points("left", fr, pts_l)
            add_bounds_points("right", fr, pts_r)
            # global
            if args.video_zoom == "global":
                add_bounds_points("left", fr, pts_glb)
                add_bounds_points("right", fr, pts_glb)

        if pts_l:
            mn_l, mx_l = _compute_bounds(pts_l)
        if pts_r:
            mn_r, mx_r = _compute_bounds(pts_r)
        if pts_glb:
            mn_glb, mx_glb = _compute_bounds(pts_glb)

        out_path = args.video_out
        if not os.path.isabs(out_path):
            out_path = os.path.abspath(out_path)

        # Render frames to a temp directory
        with tempfile.TemporaryDirectory(prefix="bvh_hand_vid_") as tmpd:
            if args.video_layout == "both":
                fig = plt.figure(figsize=(14, 7))
                ax_l = fig.add_subplot(1, 2, 1, projection="3d")
                ax_r = fig.add_subplot(1, 2, 2, projection="3d")
            else:
                fig = plt.figure(figsize=(9, 9))
                ax_l = fig.add_subplot(1, 1, 1, projection="3d")
                ax_r = None

            def plot_one(ax, pts: Dict[str, np.ndarray], end_pts: Dict[str, np.ndarray], title: str, color: str, mn: Optional[np.ndarray], mx: Optional[np.ndarray]):
                ax.cla()
                if pts:
                    P = np.stack(list(pts.values()), axis=0)
                    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=int(args.video_point_size), c=color, depthshade=True)
                if end_pts:
                    P = np.stack(list(end_pts.values()), axis=0)
                    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=int(args.video_end_point_size), c="tab:red", marker="x")
                # lines
                if args.lines and pts and parent_map:
                    for child, cpos in pts.items():
                        pj = parent_map.get(child, None)
                        if pj is None or pj not in pts:
                            continue
                        ppos = pts[pj]
                        ax.plot([ppos[0], cpos[0]], [ppos[1], cpos[1]], [ppos[2], cpos[2]], c="k", linewidth=float(args.video_line_width), alpha=0.7)
                    for ename, epos in end_pts.items():
                        j = ename.replace("_EndSite", "")
                        if j in pts:
                            jpos = pts[j]
                            ax.plot([jpos[0], epos[0]], [jpos[1], epos[1]], [jpos[2], epos[2]], c="tab:red", linewidth=float(args.video_line_width), alpha=0.8)

                ax.set_title(title)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                if float(args.video_fixed_range) > 0:
                    _apply_fixed_range(ax, float(args.video_fixed_range))
                elif mn is not None and mx is not None:
                    _apply_bounds_equal(ax, mn, mx)
                else:
                    _set_axes_equal(ax)

            for k, i in enumerate(range(vstart, vend)):
                fr = frames[i]
                l = _collect_bvh_hand_points(fr, "left")
                r = _collect_bvh_hand_points(fr, "right")
                l_end: Dict[str, np.ndarray] = {}
                r_end: Dict[str, np.ndarray] = {}
                if args.end_sites and endsite_offsets_cm:
                    for joint in endsite_offsets_cm.keys():
                        p = _compute_end_site_pos_for_joint(fr, joint, endsite_offsets_cm)
                        if p is None:
                            continue
                        if joint.startswith("Left") and (("Hand" in joint) or ("Finger" in joint)):
                            l_end[joint + "_EndSite"] = p
                        if joint.startswith("Right") and (("Hand" in joint) or ("Finger" in joint)):
                            r_end[joint + "_EndSite"] = p

                # Coordinate transform for visibility
                scale = float(args.video_scale)
                if args.video_coords == "wrist_local":
                    # Use Hand joint as origin
                    o_l = _get_pos(fr, "LeftHand") if ("LeftHand" in fr) else None
                    o_r = _get_pos(fr, "RightHand") if ("RightHand" in fr) else None
                else:
                    o_l = None
                    o_r = None

                l_t, l_end_t = _transform_points(l, l_end, o_l, scale)
                r_t, r_end_t = _transform_points(r, r_end, o_r, scale)

                if args.video_zoom == "global":
                    zl_mn, zl_mx = mn_glb, mx_glb
                    zr_mn, zr_mx = mn_glb, mx_glb
                else:
                    zl_mn, zl_mx = mn_l, mx_l
                    zr_mn, zr_mx = mn_r, mx_r

                if args.video_layout == "both":
                    plot_one(ax_l, l_t, l_end_t, f"Left hand frame={i} ({args.video_coords}, x{scale:g})", "tab:blue", zl_mn, zl_mx)
                    plot_one(ax_r, r_t, r_end_t, f"Right hand frame={i} ({args.video_coords}, x{scale:g})", "tab:orange", zr_mn, zr_mx)
                    fig.suptitle(f"BVH hands: {os.path.basename(bvh_path)}", fontsize=12)
                elif args.video_layout == "left":
                    plot_one(ax_l, l_t, l_end_t, f"Left hand frame={i} ({args.video_coords}, x{scale:g})", "tab:blue", zl_mn, zl_mx)
                    fig.suptitle(f"BVH left hand: {os.path.basename(bvh_path)}", fontsize=12)
                else:
                    # right
                    plot_one(ax_l, r_t, r_end_t, f"Right hand frame={i} ({args.video_coords}, x{scale:g})", "tab:orange", zr_mn, zr_mx)
                    fig.suptitle(f"BVH right hand: {os.path.basename(bvh_path)}", fontsize=12)
                fig.tight_layout()

                img_path = os.path.join(tmpd, f"frame_{k:06d}.png")
                fig.savefig(img_path, dpi=int(args.video_dpi))
                if (k % 30) == 0:
                    print(f"[video] rendered {k}/{(vend - vstart)} frames...")

            plt.close(fig)

            # Compose video via ffmpeg
            ffmpeg = "ffmpeg"
            ffmpeg_cmd = [
                ffmpeg,
                "-y",
                "-framerate",
                str(float(args.video_fps)),
                "-i",
                os.path.join(tmpd, "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                out_path,
            ]
            try:
                subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except FileNotFoundError:
                print("❌ 找不到 ffmpeg，请先安装 ffmpeg（或把图像序列手动合成视频）", file=sys.stderr)
                print(f"   临时帧图已生成于: {tmpd}", file=sys.stderr)
                return 4
            except subprocess.CalledProcessError as e:
                print("❌ ffmpeg 合成失败：", file=sys.stderr)
                try:
                    print(e.stderr.decode("utf-8", errors="ignore"), file=sys.stderr)
                except Exception:
                    pass
                return 4

        print(f"✅ 视频已保存: {out_path}")
        return 0

    idx = int(args.frame)
    if idx < 0 or idx >= len(frames):
        print(f"❌ frame 超界: {idx} (0..{len(frames)-1})", file=sys.stderr)
        return 2

    frame = frames[idx]
    left = _collect_bvh_hand_points(frame, "left")
    right = _collect_bvh_hand_points(frame, "right")

    # If requested, compute End Site 3D points (in the same coordinate system as loader output)
    left_end: Dict[str, np.ndarray] = {}
    right_end: Dict[str, np.ndarray] = {}
    if args.end_sites and endsite_offsets_cm:
        # loader coordinate transform (must match replay loader)
        rot_m = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        for joint, off_cm in endsite_offsets_cm.items():
            if joint not in frame:
                continue
            try:
                jpos = _get_pos(frame, joint)
                jq = _get_quat_wxyz(frame, joint)
            except Exception:
                continue
            # cm -> m, and rotate offset vector to match loader coordinate convention
            off_m = (off_cm / 100.0) @ rot_m.T
            delta = _rotate_vec_by_quat_wxyz(off_m, jq)
            end_pos = jpos + delta.astype(np.float32)
            # 注意：真正带 End Site 的往往是 Finger 的最后一节（例如 RightIndexFinger3），不含 "Hand" 字样
            if joint.startswith("Left") and (("Hand" in joint) or ("Finger" in joint)):
                left_end[joint + "_EndSite"] = end_pos
            if joint.startswith("Right") and (("Hand" in joint) or ("Finger" in joint)):
                right_end[joint + "_EndSite"] = end_pos

        print(f"[end_sites] parsed offsets: {len(endsite_offsets_cm)}, plotted left={len(left_end)} right={len(right_end)}")

    left26 = right26 = None
    if args.overlay_tracking26:
        left26 = _collect_tracking26_points(frame, "left")
        right26 = _collect_tracking26_points(frame, "right")

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"❌ 缺少 matplotlib，无法可视化：{e}", file=sys.stderr)
        print("   解决：在你的环境里安装 matplotlib，例如：pip install matplotlib", file=sys.stderr)
        return 3

    fig = plt.figure(figsize=(12, 6))
    ax_l = fig.add_subplot(1, 2, 1, projection="3d")
    ax_r = fig.add_subplot(1, 2, 2, projection="3d")

    def plot_pts(ax, pts: Dict[str, np.ndarray], title: str, color: str):
        if not pts:
            ax.set_title(title + " (no points)")
            return
        P = np.stack(list(pts.values()), axis=0)
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=30, c=color, depthshade=True)
        if args.labels:
            for name, p in pts.items():
                ax.text(p[0], p[1], p[2], name, fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    plot_pts(ax_l, left, f"Left hand (BVH joints) frame={idx}", "tab:blue")
    plot_pts(ax_r, right, f"Right hand (BVH joints) frame={idx}", "tab:orange")

    # End Sites: plot as smaller red markers and optional labels
    if left_end:
        P = np.stack(list(left_end.values()), axis=0)
        ax_l.scatter(P[:, 0], P[:, 1], P[:, 2], s=18, c="tab:red", marker="x")
        if args.labels:
            for name, p in left_end.items():
                ax_l.text(p[0], p[1], p[2], name, fontsize=7, color="tab:red")
    if right_end:
        P = np.stack(list(right_end.values()), axis=0)
        ax_r.scatter(P[:, 0], P[:, 1], P[:, 2], s=18, c="tab:red", marker="x")
        if args.labels:
            for name, p in right_end.items():
                ax_r.text(p[0], p[1], p[2], name, fontsize=7, color="tab:red")

    # Skeleton lines: connect parent-child within hand joints, plus joint->EndSite
    def plot_lines(ax, pts: Dict[str, np.ndarray], end_pts: Dict[str, np.ndarray]):
        if not pts or not parent_map:
            return
        # joint-joint
        for child, cpos in pts.items():
            pj = parent_map.get(child, None)
            if pj is None or pj not in pts:
                continue
            ppos = pts[pj]
            ax.plot([ppos[0], cpos[0]], [ppos[1], cpos[1]], [ppos[2], cpos[2]], c="k", linewidth=1.0, alpha=0.6)
        # joint -> end site
        for ename, epos in end_pts.items():
            joint = ename.replace("_EndSite", "")
            if joint in pts:
                jpos = pts[joint]
                ax.plot([jpos[0], epos[0]], [jpos[1], epos[1]], [jpos[2], epos[2]], c="tab:red", linewidth=1.0, alpha=0.7)

    if args.lines:
        plot_lines(ax_l, left, left_end)
        plot_lines(ax_r, right, right_end)

    if left26 is not None:
        P = np.stack(list(left26.values()), axis=0) if left26 else None
        if P is not None:
            ax_l.scatter(P[:, 0], P[:, 1], P[:, 2], s=12, c="tab:green", alpha=0.8)
    if right26 is not None:
        P = np.stack(list(right26.values()), axis=0) if right26 else None
        if P is not None:
            ax_r.scatter(P[:, 0], P[:, 1], P[:, 2], s=12, c="tab:green", alpha=0.8)

    # Try to make both plots comparable
    _set_axes_equal(ax_l)
    _set_axes_equal(ax_r)
    fig.suptitle(f"BVH hands visualization: {os.path.basename(bvh_path)}", fontsize=12)
    fig.tight_layout()

    if args.out:
        out_path = args.out
        if not os.path.isabs(out_path):
            out_path = os.path.abspath(out_path)
        fig.savefig(out_path, dpi=200)
        print(f"✅ 已保存: {out_path}")

    if not args.no_show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


