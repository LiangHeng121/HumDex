#!/usr/bin/env python3
"""
测试 Wuji 手设备访问
"""
import sys

print("=" * 60)
print("Wuji 手设备访问测试")
print("=" * 60)

# 检查设备文件
import os
device_path = "/dev/ttyACM0"
print(f"\n1. 检查设备文件: {device_path}")
if os.path.exists(device_path):
    stat = os.stat(device_path)
    print(f"   ✅ 设备存在")
    print(f"   权限: {oct(stat.st_mode)}")
    print(f"   所有者: {stat.st_uid}:{stat.st_gid}")
    
    # 尝试读取权限
    if os.access(device_path, os.R_OK):
        print(f"   ✅ 可读")
    else:
        print(f"   ❌ 不可读")
    
    if os.access(device_path, os.W_OK):
        print(f"   ✅ 可写")
    else:
        print(f"   ❌ 不可写")
else:
    print(f"   ❌ 设备不存在")
    sys.exit(1)

# 检查 wujihandpy
print(f"\n2. 检查 wujihandpy 库")
try:
    import wujihandpy
    print(f"   ✅ wujihandpy 已安装")
    print(f"   版本: {getattr(wujihandpy, '__version__', '未知')}")
except ImportError as e:
    print(f"   ❌ wujihandpy 未安装: {e}")
    sys.exit(1)

# 尝试初始化
print(f"\n3. 尝试初始化 Wuji 手")
try:
    print("   正在初始化...")
    hand = wujihandpy.Hand()
    print("   ✅ 初始化成功！")
    
    # 尝试获取位置
    print("   尝试获取关节位置...")
    pos = hand.get_joint_actual_position()
    print(f"   ✅ 成功获取位置: shape={pos.shape}")
    
    # 清理
    hand.write_joint_enabled(False)
    print("   ✅ 设备访问正常")
    
except RuntimeError as e:
    print(f"   ❌ 初始化失败: {e}")
    print("\n   可能的解决方案:")
    print("   1. 重新插拔 USB 设备")
    print("   2. 检查设备是否被其他程序占用")
    print("   3. 尝试重启系统")
    print("   4. 检查 wujihandpy 库版本是否兼容")
    sys.exit(1)
except Exception as e:
    print(f"   ❌ 发生错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有测试通过！")
print("=" * 60)

