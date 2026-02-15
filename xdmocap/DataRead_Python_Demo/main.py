from vdmocapsdk_dataread import *
from vdmocapsdk_nodelist import *

global_index = 0
global_dst_ip = "192.168.31.134"    # 广播 IP 请根据广播设置自行修改
global_dst_port = 7000    # 广播端口请根据广播设置自行修改
global_local_port = 0
global_world_space = 0
# 下面三个骨架是地理坐标系下的身体、右手和左手的关节坐标数据
global_initial_position_body = [
        [0, 0, 1.022],
        [0.074, 0, 1.002],
        [0.097, 0, 0.593],
        [0.104, 0, 0.111],
        [0.114, 0.159, 0.005],
        [-0.074, 0, 1.002],
        [-0.097, 0.001, 0.593],
        [-0.104, 0, 0.111],
        [-0.114, 0.158, 0.004],
        [0, 0.033, 1.123],
        [0, 0.03, 1.246],
        [0, 0.014, 1.362],
        [0, -0.048, 1.475],
        [0, -0.048, 1.549],
        [0, -0.016, 1.682],
        [0.071, -0.061, 1.526],
        [0.178, -0.061, 1.526],
        [0.421, -0.061, 1.526],
        [0.682, -0.061, 1.526],
        [-0.071, -0.061, 1.526],
        [-0.178, -0.061, 1.526],
        [-0.421, -0.061, 1.526],
        [-0.682, -0.061, 1.526],
]
global_initial_position_hand_right = [
        [0.682, -0.061, 1.526],
        [0.71, -0.024, 1.526],
        [0.728, -0.008, 1.526],
        [0.755, 0.013, 1.526],
        [0.707, -0.05, 1.526],
        [0.761, -0.024, 1.525],
        [0.812, -0.023, 1.525],
        [0.837, -0.022, 1.525],
        [0.709, -0.058, 1.526],
        [0.764, -0.046, 1.528],
        [0.816, -0.046, 1.528],
        [0.845, -0.046, 1.528],
        [0.709, -0.064, 1.526],
        [0.761, -0.069, 1.527],
        [0.812, -0.069, 1.527],
        [0.835, -0.069, 1.527],
        [0.708, -0.072, 1.526],
        [0.755, -0.089, 1.522],
        [0.791, -0.089, 1.522],
        [0.81, -0.089, 1.522],
    ]
global_initial_position_hand_left = [
        [-0.682, -0.061, 1.526],
        [-0.71, -0.024, 1.526],
        [-0.728, -0.008, 1.526],
        [-0.755, 0.013, 1.526],
        [-0.707, -0.05, 1.526],
        [-0.761, -0.024, 1.525],
        [-0.812, -0.023, 1.525],
        [-0.837, -0.022, 1.525],
        [-0.709, -0.058, 1.526],
        [-0.764, -0.046, 1.528],
        [-0.816, -0.046, 1.528],
        [-0.845, -0.046, 1.528],
        [-0.709, -0.064, 1.526],
        [-0.761, -0.069, 1.527],
        [-0.812, -0.069, 1.527],
        [-0.835, -0.069, 1.527],
        [-0.708, -0.072, 1.526],
        [-0.755, -0.089, 1.522],
        [-0.791, -0.089, 1.522],
        [-0.81, -0.089, 1.522],
    ]


def Test():
    global global_index
    global global_dst_ip
    global global_dst_port
    global global_local_port
    global global_world_space
    global global_initial_position_body
    global global_initial_position_hand_right
    global global_initial_position_hand_left

    mocap_data = MocapData()

    # 打开 UDP
    if not udp_is_open(global_index):
        if not udp_open(global_index, global_local_port):
            return
    # 设置重定向骨架
    udp_set_position_in_initial_tpose(global_index, global_dst_ip, global_dst_port,
                                             global_world_space, global_initial_position_body,
                                             global_initial_position_hand_right,
                                             global_initial_position_hand_left)
    # 请求 UDP 连接
    if not udp_send_request_connect(global_index, global_dst_ip, global_dst_port):
        return
    for k in range(100):
        # 前面设置了重定向骨架后，接口 udp_recv_mocap_data() 就能获取根据骨架进行重定向后的数据
        if udp_recv_mocap_data(global_index, global_dst_ip, global_dst_port, mocap_data):
            if mocap_data.isUpdate:
                info = "\n==================== 动捕数据 ====================\r\n"
                info += f'frame index: {mocap_data.frameIndex}\n'
                info += "===================  身体数据  ===================\n"
                for i in range(LENGTH_BODY):
                    for j in range(3):
                        info += '%f '%(mocap_data.position_body[i][j])
                    info += "  -   "
                    for j in range(4):
                        info += '%f '%(mocap_data.quaternion_body[i][j])
                    info += "\n"
                info += "===================  右手数据  ===================\n"
                for i in range(LENGTH_HAND):
                    for j in range(3):
                        info += '%f '%(mocap_data.position_rHand[i][j])
                    info += "  -   "
                    for j in range(4):
                        info += '%f '%(mocap_data.quaternion_rHand[i][j])
                    info += "\n"
                info += "===================  左手数据  ===================\n"
                for i in range(LENGTH_HAND):
                    for j in range(3):
                        info += '%f '%(mocap_data.position_lHand[i][j])
                    info += "  -   "
                    for j in range(4):
                        info += '%f '%(mocap_data.quaternion_lHand[i][j])
                    info += "\n"
                info += "===================  表情数据  ===================\n"
                for i in range(LENGTH_FACE):
                    if i >= 9 and (i + 1) % 10 == 1:
                        info += "\n"
                    info += f'{mocap_data.faceBlendShapesARKit[i]} '
                info += "\n"

                print(info)
                
        sleep(0.002)
        
    # 关闭 UDP 连接
    if udp_remove(global_index, global_dst_ip, global_dst_port):
        udp_close(global_index)


if __name__ == "__main__":
    Test()        
