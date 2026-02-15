python3 /home/heng/heng/G1/TWIST2/deploy_real/scripts/monitor_action_body_vs_reference.py \
  --ref_dir /home/heng/heng/G1/TWIST2/deploy_real/twist2_demonstration/20260117_2127_robot1 \
  --ref_episode episode_0000 \
  --ref_indices "0,207,329" \
  --redis_ip localhost \
  --fk \
  --eef_thr_m 0.06

#   按键：
# n：进入下一参考帧
# p：打印一次当前详细误差
# q：退出
# > 说明：如果你加了 --fk 但本环境没有 mujoco，脚本会直接报错退出，避免“末端差”被悄悄跳过。

# 输入参考机器人数据目录（session/episode/data.json 都支持）+ n 个参考帧 idx
# 循环读取 Redis 当前 action_body_*（默认 action_body_unitree_g1_with_hands，35D mimic_obs）
# 与当前参考帧对比：
# 关节差（取 mimic_obs 的后 29D dof_pos，计算 joint_linf / joint_rms）
# 手臂末端差（可选：用 MuJoCo FK 计算 left_rubber_hand/right_rubber_hand 的位置差）
# 当差异都小于阈值时提示你，并且你按键后进入下一参考帧


# 123 93 138

# robot 62cm