#!/bin/bash
# Wuji 手设备权限设置脚本

echo "=========================================="
echo "Wuji 手设备权限设置"
echo "=========================================="

# 创建 udev 规则
echo "创建 udev 规则..."
sudo tee /etc/udev/rules.d/95-wujihand.rules > /dev/null << 'EOF'
# WujiHand USB 设备权限规则
# USB 设备权限（wujihandpy 库直接访问 USB）
SUBSYSTEM=="usb", ATTR{idVendor}=="0483", ATTR{idProduct}=="2000", MODE="0666", GROUP="dialout"

# TTY 设备权限（串口通信）
SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="2000", MODE="0666", GROUP="dialout"
EOF

echo "✅ udev 规则已创建"

# 重新加载 udev 规则
echo "重新加载 udev 规则..."
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "✅ udev 规则已重新加载"

# 设置当前设备权限
echo "设置当前设备权限..."
if [ -e /dev/ttyACM0 ]; then
    sudo chmod a+rw /dev/ttyACM0
    echo "✅ /dev/ttyACM0 权限已设置"
else
    echo "⚠️  /dev/ttyACM0 不存在，请检查设备连接"
fi

# 检查 USB 设备节点
USB_DEVICE=$(lsusb -d 0483:2000 | awk '{print $2"/"$4}' | sed 's/://')
if [ -n "$USB_DEVICE" ]; then
    USB_PATH="/dev/bus/usb/$USB_DEVICE"
    if [ -e "$USB_PATH" ]; then
        sudo chmod a+rw "$USB_PATH"
        echo "✅ $USB_PATH 权限已设置"
    fi
fi

echo ""
echo "=========================================="
echo "✅ 权限设置完成！"
echo ""
echo "如果设备仍然无法访问，请："
echo "1. 重新插拔 USB 设备"
echo "2. 或者重启系统"
echo "=========================================="

