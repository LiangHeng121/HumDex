## 用 Docker（GPU）打包当前 TWIST2 conda 环境

### 1) 在宿主机导出你当前的 conda 环境

在仓库根目录：

```bash
cd /home/heng/heng/G1/TWIST2
conda env export -n twist2 --no-builds > environment.twist2.yml
```

说明：
- `docker/Dockerfile.gpu` 会在构建时自动去掉 `prefix:`，并把 `channels` 归一成 `conda-forge` + `defaults`（避免写死清华镜像 URL）。

### 2) 构建镜像

```bash
cd /home/heng/heng/G1/TWIST2
docker build -f docker/Dockerfile.gpu -t twist2:gpu .
```

### 3) 运行（需要 GPU）

离线跑（例如 `--sim_only --sim_save_vid` 保存视频，不需要 GUI）：

```bash
cd /home/heng/heng/G1/TWIST2
docker run --rm -it --gpus all \
  -v "$PWD:/workspace/TWIST2" \
  --net host \
  twist2:gpu
```

进入容器后就直接用：

```bash
python deploy_real/policy_inference.py replay --help
```

### 4) 常见注意事项

- 如果你需要连 Redis（例如 `infer` 模式），建议使用 `--net host`（或自己映射端口）。
- 如果你后续需要弹窗（OpenCV/MuJoCo viewer），需要额外做 X11/Wayland 转发；离线保存 mp4 不需要。


