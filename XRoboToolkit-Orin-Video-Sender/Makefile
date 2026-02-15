###############################################################################
#
# Standalone Makefile for OrinVideoSender
# Self-contained build configuration
# By liuchuan.yu@bytedance.com
# Modified for Unitree G1 (RealSense/Webcam Mode)
#
###############################################################################

# Compiler settings
CXX = g++
APP := OrinVideoSender

###############################################################################
# WebCam (RealSense 走这个模式)

# TCP w/o asio -- pass
SRCS := \
	main_web_gst.cpp
###############################################################################

###############################################################################
# ZED (G1通常没装ZED SDK，这里的都被注释掉了)

# TCP w/o asio -- pass
# SRCS := \
#     main_zed_tcp.cpp

# SRCS := \
#     main_zed_tcp_zmq.cpp

# # TCP with asio -- pass
# SRCS := \
#   main_zed_asio.cpp

# # UDP w/ asio -- pass
# SRCS:= \
#   main_zed_asio_udp.cpp

# # [NOT WORKING] Zero Copy - depends on jetson multimedia api
# SRCS := \
#   main_zed_zero_copy.cpp
###############################################################################

OBJS := $(SRCS:.cpp=.o)

# Include paths
# [修改] 去掉了 -I/usr/local/zed/include 防止报错
CPPFLAGS := -std=c++11 \
	-I./asio-1.30.2/include \
	-I/usr/include/opencv4 \
	-I/usr/local/cuda/include \
	$(shell pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 glib-2.0 2>/dev/null || echo "") \
	$(shell pkg-config --cflags libzmq 2>/dev/null || echo "")

# Compiler flags
CXXFLAGS := -Wall -Wextra -O2 -g

# FFmpeg flags
CXXFLAGS += $(shell pkg-config --cflags libavcodec libavformat libavutil libswscale libavdevice 2>/dev/null || echo "")

# Library paths and libraries
# [修改] 去掉了 -L/usr/local/zed/lib
LDFLAGS := -L/usr/local/cuda/lib64
	# -L/usr/lib/aarch64-linux-gnu

# FFmpeg libraries (must come first to avoid conflicts)
LDFLAGS += $(shell pkg-config --libs libavcodec libavformat libavutil libswscale libavdevice 2>/dev/null || echo "-lavcodec -lavformat -lavutil -lswscale -lavdevice")

# Core libraries
# [修改] 去掉了 -lsl_zed (这是ZED专用库，没有它会导致编译失败)
LDFLAGS += -lcuda -lcudart \
	-lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs \
	-lssl -lcrypto \
	-lpthread \
	-lstdc++

# GStreamer libraries
LDFLAGS += $(shell pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 glib-2.0 2>/dev/null || echo "-lgstreamer-1.0 -lgstapp-1.0 -lglib-2.0")

# ZMQ library
LDFLAGS += $(shell pkg-config --libs libzmq 2>/dev/null || echo "-lzmq")

all: $(APP)

debug: CXXFLAGS += -DDEBUG -g3 -O0
debug: $(APP)

%.o: %.cpp
	@echo "Compiling: $<"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(APP): $(OBJS)
	@echo "Linking: $@"
	$(CXX) -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -rf $(APP) $(OBJS)

install: $(APP)
	@echo "Installing $(APP)..."
	install -D $(APP) /usr/local/bin/$(APP)

.PHONY: all debug clean install