#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

// 系统网络库
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>

// GStreamer
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <glib-unix.h>
#include <glib.h>

#include "network_helper.hpp"

// ZMQ（用于“同一份相机流”同时输出：VR(TCP) + 数据采集(ZMQ)）
#include <zmq.h>

// ================= 全局变量 =================
std::unique_ptr<TCPClient> sender_ptr; // 视频发送通道 (To VR 12345)
GMainLoop *loop = nullptr;
volatile sig_atomic_t stop_requested = 0;
bool send_enabled = false; 

// ZMQ：编码流(H.264) / 原始图像(BGR/BGRA)
std::atomic<bool> zmq_enabled(false);
std::atomic<bool> zmq_raw_enabled(false);
void* zmq_context = nullptr;
void* zmq_publisher = nullptr;
std::string zmq_endpoint = "";
std::mutex zmq_mutex;

// 线程同步变量
std::string target_vr_ip = "";
std::atomic<bool> vr_connected(false); // VR是否连上了控制端口
std::mutex ip_mutex;

// ================= 信号处理 =================
static void signal_handler(int sig) {
  if (!stop_requested && loop) {
    stop_requested = 1;
    g_main_loop_quit(loop);
  }
}

void printErrorAndQuit(const std::string &error_msg) {
  std::cerr << "Error: " << error_msg << std::endl;
}

static bool initialize_zmq(const std::string& endpoint) {
  if (endpoint.empty()) return false;
  std::lock_guard<std::mutex> lock(zmq_mutex);

  if (zmq_publisher || zmq_context) return true;

  zmq_context = zmq_ctx_new();
  if (!zmq_context) {
    std::cerr << "Failed to create ZMQ context" << std::endl;
    return false;
  }

  zmq_publisher = zmq_socket(zmq_context, ZMQ_PUB);
  if (!zmq_publisher) {
    std::cerr << "Failed to create ZMQ publisher socket" << std::endl;
    zmq_ctx_term(zmq_context);
    zmq_context = nullptr;
    return false;
  }

  int hwm = 1;         // 丢帧优先，保证低延迟
  int linger = 0;      // 退出时不阻塞
  zmq_setsockopt(zmq_publisher, ZMQ_SNDHWM, &hwm, sizeof(hwm));
  zmq_setsockopt(zmq_publisher, ZMQ_LINGER, &linger, sizeof(linger));

  if (zmq_bind(zmq_publisher, endpoint.c_str()) != 0) {
    std::cerr << "Failed to bind ZMQ publisher to " << endpoint
              << " : " << zmq_strerror(zmq_errno()) << std::endl;
    zmq_close(zmq_publisher);
    zmq_publisher = nullptr;
    zmq_ctx_term(zmq_context);
    zmq_context = nullptr;
    return false;
  }

  std::cout << "ZMQ publisher bound to " << endpoint << std::endl;
  return true;
}

static void cleanup_zmq() {
  std::lock_guard<std::mutex> lock(zmq_mutex);
  if (zmq_publisher) {
    zmq_close(zmq_publisher);
    zmq_publisher = nullptr;
  }
  if (zmq_context) {
    zmq_ctx_term(zmq_context);
    zmq_context = nullptr;
  }
}

// ================= GStreamer 回调：编码 H264（发 VR + 可选 ZMQ） =================
GstFlowReturn on_new_sample_encoded(GstAppSink *sink, gpointer user_data) {
  GstSample *sample = gst_app_sink_pull_sample(sink);
  if (!sample) return GST_FLOW_ERROR;

  GstBuffer *buffer = gst_sample_get_buffer(sample);
  GstMapInfo map;
  
  if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    const uint8_t *data = map.data;
    gsize size = map.size;
    
    // 只有当 send_enabled 为真且连接正常时才发送
    if (send_enabled && sender_ptr && sender_ptr->isConnected() && data && size > 0) {
      try {
        std::vector<uint8_t> packet(4 + size);
        packet[0] = (size >> 24) & 0xFF;
        packet[1] = (size >> 16) & 0xFF;
        packet[2] = (size >> 8) & 0xFF;
        packet[3] = (size)&0xFF;
        std::copy(data, data + size, packet.begin() + 4);
        sender_ptr->sendData(packet);
      } catch (const std::exception &e) {
        // 发送失败不退出，只是打印警告
        // std::cerr << "发送数据包失败 (网络波动?)" << std::endl;
      }
    }

    // 同时通过 ZMQ 发送编码流（格式：[4字节len][H264 bytes...]）
    if (zmq_enabled.load() && zmq_publisher && data && size > 0) {
      try {
        std::lock_guard<std::mutex> lock(zmq_mutex);
        std::vector<uint8_t> zmq_packet(4 + size);
        zmq_packet[0] = (size >> 24) & 0xFF;
        zmq_packet[1] = (size >> 16) & 0xFF;
        zmq_packet[2] = (size >> 8) & 0xFF;
        zmq_packet[3] = (size) & 0xFF;
        std::copy(data, data + size, zmq_packet.begin() + 4);
        int rc = zmq_send(zmq_publisher, zmq_packet.data(), zmq_packet.size(), ZMQ_DONTWAIT);
        if (rc == -1 && zmq_errno() != EAGAIN) {
          std::cerr << "ZMQ send error (h264): " << zmq_strerror(zmq_errno()) << std::endl;
        }
      } catch (const std::exception &e) {
        std::cerr << "Unexpected error in ZMQ send (h264): " << e.what() << std::endl;
      }
    }

    gst_buffer_unmap(buffer, &map);
  }
  gst_sample_unref(sample);
  return GST_FLOW_OK;
}

// ================= GStreamer 回调：原始图像（ZMQ-RAW） =================
GstFlowReturn on_new_sample_raw(GstAppSink *sink, gpointer user_data) {
  GstSample *sample = gst_app_sink_pull_sample(sink);
  if (!sample) return GST_FLOW_ERROR;

  if (!zmq_raw_enabled.load() || !zmq_publisher) {
    gst_sample_unref(sample);
    return GST_FLOW_OK;
  }

  GstCaps *caps = gst_sample_get_caps(sample);
  int width = 0, height = 0;
  int channels = 3;
  if (caps) {
    GstStructure *s = gst_caps_get_structure(caps, 0);
    gst_structure_get_int(s, "width", &width);
    gst_structure_get_int(s, "height", &height);
    const gchar *fmt = gst_structure_get_string(s, "format");
    if (fmt) {
      if (std::string(fmt) == "BGRx" || std::string(fmt) == "BGRA") channels = 4;
      else channels = 3;
    }
  }

  GstBuffer *buffer = gst_sample_get_buffer(sample);
  GstMapInfo map;
  if (buffer && gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    const uint8_t *data = map.data;
    size_t size = map.size;

    if (data && size > 0 && width > 0 && height > 0) {
      // 格式：[4字节width][4字节height][4字节channels][raw bytes...]
      try {
        std::lock_guard<std::mutex> lock(zmq_mutex);
        std::vector<uint8_t> pkt(12 + size);
        std::memcpy(&pkt[0], &width, 4);
        std::memcpy(&pkt[4], &height, 4);
        std::memcpy(&pkt[8], &channels, 4);
        std::memcpy(&pkt[12], data, size);
        int rc = zmq_send(zmq_publisher, pkt.data(), pkt.size(), ZMQ_DONTWAIT);
        if (rc == -1 && zmq_errno() != EAGAIN) {
          std::cerr << "ZMQ send error (raw): " << zmq_strerror(zmq_errno()) << std::endl;
        }
      } catch (const std::exception &e) {
        std::cerr << "Unexpected error in ZMQ send (raw): " << e.what() << std::endl;
      }
    }

    gst_buffer_unmap(buffer, &map);
  }

  gst_sample_unref(sample);
  return GST_FLOW_OK;
}

// ================= 命令服务器线程 (保持 13579 连接) =================
void command_server_thread_func(int port) {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    std::cout << ">>> [命令线程] 启动监听端口 " << port << "..." << std::endl;

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Socket failed");
        return;
    }
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
    
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        return;
    }
    if (listen(server_fd, 3) < 0) {
        perror("Listen");
        return;
    }

    while (!stop_requested) {
        // 接受连接
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
            if (stop_requested) break;
            usleep(100000); 
            continue;
        }

        // 获取 VR IP
        char vr_ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(address.sin_addr), vr_ip_str, INET_ADDRSTRLEN);
        
        {
            std::lock_guard<std::mutex> lock(ip_mutex);
            target_vr_ip = std::string(vr_ip_str);
            vr_connected = true; 
        }

        std::cout << ">>> [握手成功!] VR (" << target_vr_ip << ") 已连接控制通道 13579!" << std::endl;

        // 【死循环读取】保持连接不挂断，直到对方断开
        char buffer[1024] = {0};
        while (!stop_requested) {
            int valread = read(new_socket, buffer, 1024);
            if (valread <= 0) {
                std::cout << ">>> VR 断开了控制连接" << std::endl;
                close(new_socket);
                vr_connected = false;
                // 重置发送状态，等待下一次重连
                send_enabled = false;
                break;
            }
            // 收到指令不处理，只打印，保持连接活跃
            // std::cout << ">>> 收到心跳/指令" << std::endl;
        }
    }
    close(server_fd);
}

// ================= 主函数 =================
int main(int argc, char *argv[]) {
  gst_init(&argc, &argv);
  
  bool listen_mode = false;
  int cmd_port = 13579;
  std::string zmq_arg = "";
  std::string zmq_raw_arg = "";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--listen") listen_mode = true;
    else if (arg == "--zmq" && i + 1 < argc) {
      zmq_arg = argv[++i];
    } else if (arg == "--zmq-raw" && i + 1 < argc) {
      zmq_raw_arg = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "用法：\n"
                << "  ./OrinVideoSender --listen <IP>:13579 [--preview]\n"
                << "  ./OrinVideoSender --listen <IP>:13579 --zmq tcp://*:5555        # ZMQ发H264\n"
                << "  ./OrinVideoSender --listen <IP>:13579 --zmq-raw tcp://*:5556    # ZMQ发原始BGR\n";
      return 0;
    }
  }

  if (!listen_mode) {
      std::cout << "请使用: ./OrinVideoSender --listen <IP>:13579" << std::endl;
      return 0;
  }

  // 信号处理（Ctrl+C 等）
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  if (!zmq_arg.empty()) {
    zmq_endpoint = zmq_arg;
    zmq_enabled.store(true);
    if (!initialize_zmq(zmq_endpoint)) {
      std::cerr << "❌ 初始化 ZMQ 失败（--zmq " << zmq_endpoint << "）" << std::endl;
      return -2;
    }
  } else if (!zmq_raw_arg.empty()) {
    zmq_endpoint = zmq_raw_arg;
    zmq_raw_enabled.store(true);
    if (!initialize_zmq(zmq_endpoint)) {
      std::cerr << "❌ 初始化 ZMQ 失败（--zmq-raw " << zmq_endpoint << "）" << std::endl;
      return -2;
    }
  }

  // 1. 启动命令监听线程 (后台运行)
  std::thread cmd_thread(command_server_thread_func, cmd_port);
  cmd_thread.detach();

  // 2. 配置 GStreamer Pipeline (RealSense 优化版)
  // - 编码支路：用于给 VR 发 H.264
  // - 可选 raw 支路：用于 ZMQ-RAW 给数据采集直接拿到 BGR
  std::string pipeline_desc;
  if (zmq_raw_enabled.load()) {
    pipeline_desc =
        "v4l2src device=/dev/video4 ! "
        "video/x-raw, width=1280, height=720, format=YUY2, framerate=30/1 ! "
        "tee name=t "
        "t. ! queue ! "
        "nvvidconv ! video/x-raw(memory:NVMM), format=NV12 ! "
        "nvv4l2h264enc maxperf-enable=1 bitrate=2000000 insert-sps-pps=true idrinterval=15 iframeinterval=15 ! "
        "h264parse ! appsink name=encsink emit-signals=true sync=false "
        "t. ! queue ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink name=rawsink emit-signals=true sync=false";
  } else {
    pipeline_desc =
        "v4l2src device=/dev/video4 ! "
        "video/x-raw, width=1280, height=720, format=YUY2, framerate=30/1 ! "
        "nvvidconv ! video/x-raw(memory:NVMM), format=NV12 ! "
        "nvv4l2h264enc maxperf-enable=1 bitrate=2000000 insert-sps-pps=true idrinterval=15 iframeinterval=15 ! "
        "h264parse ! appsink name=encsink emit-signals=true sync=false";
  }

  GError *error = nullptr;
  GstElement *pipeline = gst_parse_launch(pipeline_desc.c_str(), &error);
  if (!pipeline) {
    g_printerr("Pipeline Error: %s\n", error->message);
    return -1;
  }
  GstElement *encsink = gst_bin_get_by_name(GST_BIN(pipeline), "encsink");
  g_signal_connect(encsink, "new-sample", G_CALLBACK(on_new_sample_encoded), nullptr);
  GstElement *rawsink = nullptr;
  if (zmq_raw_enabled.load()) {
    rawsink = gst_bin_get_by_name(GST_BIN(pipeline), "rawsink");
    g_signal_connect(rawsink, "new-sample", G_CALLBACK(on_new_sample_raw), nullptr);
  }
  
  // 先让摄像头跑起来，准备好数据
  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  loop = g_main_loop_new(nullptr, FALSE);

  // 3. 启动连接管理线程 (负责不断尝试连接 VR 视频端口)
  std::thread connect_manager([]() {
      while (!stop_requested) {
          // 等待 VR 握手成功
          if (!vr_connected) {
              usleep(500000); // 0.5秒检查一次
              continue;
          }

          std::string current_ip;
          {
              std::lock_guard<std::mutex> lock(ip_mutex);
              current_ip = target_vr_ip;
          }

          if (current_ip.empty()) continue;

          // 尝试连接 VR 视频端口
          if (!send_enabled) {
              try {
                  std::cout << ">>> [重试] 正在连接 VR 视频端口 " << current_ip << ":12345 ..." << std::endl;
                  // 重新创建 client
                  sender_ptr = std::unique_ptr<TCPClient>(new TCPClient(current_ip, 12345));
                  sender_ptr->connect();
                  
                  send_enabled = true; // 连接成功，允许发送
                  std::cout << ">>> [成功] 视频通道已打通! 画面应该出现了!" << std::endl;
              } catch (const std::exception &e) {
                  std::cerr << ">>> [失败] 无法连接 VR 视频端口 (" << e.what() << ")，1秒后重试..." << std::endl;
                  send_enabled = false;
                  sleep(1); // 失败后等待1秒再重试，绝对不退出程序！
              }
          } else {
              // 已经连接，监控连接状态，如果断了就置为 false 触发重连
              if (!sender_ptr || !sender_ptr->isConnected()) {
                   send_enabled = false;
                   std::cout << ">>> [断开] 视频连接意外断开，准备重连..." << std::endl;
              }
              sleep(2);
          }
      }
  });
  connect_manager.detach();

  // 4. 进入主循环
  std::cout << ">>> 程序启动完毕，等待 VR 连接..." << std::endl;
  g_main_loop_run(loop);

  // 清理
  if (sender_ptr) sender_ptr->disconnect();
  cleanup_zmq();
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  if (encsink) gst_object_unref(encsink);
  if (rawsink) gst_object_unref(rawsink);
  g_main_loop_unref(loop);

  return 0;
}