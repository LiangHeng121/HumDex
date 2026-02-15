#pragma once
#ifndef VDMOCAPSDK_DATAREAD_H
#define VDMOCAPSDK_DATAREAD_H

//
#ifndef VDMOCAPSDKDATAREAD_API
#ifdef VDMOCAPSDKDATAREAD_EXPORTS
#define VDMOCAPSDKDATAREAD_API __declspec(dllexport)
#else
#define VDMOCAPSDKDATAREAD_API __declspec(dllimport)
#endif
#endif

//
#include "VDMocapSDK_DataRead_DataType.h"



#ifdef __cplusplus
extern "C" {
#endif

	namespace VDDataRead
	{
		/**********************************************************************************
		* @brief
		*   获取版本信息
		***********************************************************************************/
		/**
		 * @brief
		 *   获取版本信息。
		 * @param[out] version 版本信息。
		 */
		VDMOCAPSDKDATAREAD_API void GetVersionInfo(_Version_* version);


		/**********************************************************************************
		* @brief
		*   使用udp通讯协议，读取并输出数据
		***********************************************************************************/
		/**
		* @brief
		*   打开本地端口。
		* @param[in] index 对象编号（提供了多对象无关联且同时调用该函数的可行性）。
		* @return Ture,成功；false, 失败。
		*/
		VDMOCAPSDKDATAREAD_API bool UdpOpen(int index, unsigned short localPort);

		/**
		* @brief
		*   关闭由"Open"函数打开的本地端口。
		* @param[in] index 对象编号（提供了多对象无关联且同时调用该函数的可行性）。
		*/
		VDMOCAPSDKDATAREAD_API void UdpClose(int index);

		/**
		* @brief
		*   判断 udp 本地端口是否打开
		* @param[in] index 对象编号（提供了多对象无关联且同时调用该函数的可行性）。
		*/
		VDMOCAPSDKDATAREAD_API bool UdpIsOpen(int index);

		/**
		* @brief
		*   移除已连接的远程端。
		* @param[in] index 对象编号（提供了多对象无关联且同时调用该函数的可行性）。
		* @param[in] dst_ip 远程端ip地址。
		* @param[in] dst_port 远程端端口。
		* @return True, 成功；false, 失败。
		*/
		VDMOCAPSDKDATAREAD_API bool UdpRemove(int index, const char* dst_ip, unsigned short dst_port);

		/**
		* @brief
		*   设置 Tpose 骨架，若不设置，则默认使用读取到的骨架。
		* @param[in] index 对象编号（提供了多对象无关联且同时调用该函数的可行性）。
		* @param[in] dst_ip 远程端ip地址。
		* @param[in] dst_port 远程端端口。
		* @param[in] worldSpace 模型所在的世界坐标系。
		* @param[in] initialPosition 模型初始Tpose下各节点坐标，且按照枚举"_BodyNodes_"来排序。
		* @param[in] initialPosition_rHand 模型初始Tpose下右手各节点坐标，且按照枚举"_HandNodes_"来排序。
		* @param[in] initialPosition_lHand 模型初始Tpose下左手各节点坐标，且按照枚举"_HandNodes_"来排序。
		* @return Ture, 成功；false, 失败。
		* @remark
		*   (1) UdpOpen()前后均可进行设置。
		*   (1) 枚举"_BodyNodes_"及"_HandNodes_"在头文件"VDMocapSDK_DataRead_DataType.h"中定义。
		*   (2) 所述T_pose：模型站立，双腿平行，且双脚脚掌都指向正前方，左右手分别向左右平举，掌心向下。
		*/
		VDMOCAPSDKDATAREAD_API bool UdpSetPositionInInitialTpose(int index, const char* dst_ip, unsigned short dst_port, _WorldSpace_ worldSpace,
			float initialPosition_body[NODES_BODY][3], float initialPosition_rHand[NODES_HAND][3], float initialPosition_lHand[NODES_HAND][3]);

		/**
		* @brief
		*   发送连接请求。
		* @param[in] index 对象编号（提供了多对象无关联且同时调用该函数的可行性）。
		* @param[in] dst_ip 远程端ip地址。
		* @param[in] dst_port 远程端端口。
		* @return Ture, 成功；false, 失败。
		*/
		VDMOCAPSDKDATAREAD_API bool UdpSendRequestConnect(int index, const char* dst_ip, unsigned short dst_port);

		/**
		* @brief
		*   获取远程端发送过来的动捕数据。
		*   若未设置骨架，则得到的是地理坐标系下的数据。
		* @param[in] index 对象编号（提供了多对象无关联且同时调用该函数的可行性）。
		* @param[in] dst_ip 远程端ip地址。
		* @param[in] dst_port 远程端端口。
		* @param[out] mocapData 动捕数据。
		* @return Ture, 成功；false, 失败。
		*   结构体"_MocapData_"在头文件"VDMocapSDK_DataRead_DataType.h"中定义。
		*/
		VDMOCAPSDKDATAREAD_API bool UdpRecvMocapData(int index, const char* dst_ip, unsigned short dst_port, _MocapData_* mocapData);


		/**
		* @brief
		*   获取由udp接收到的模型初始Tpose各节点坐标。
		* @param[in] index 对象编号（提供了多对象无关联且同时调用该函数的可行性）。
		* @param[in] dst_ip 远程端ip地址。
		* @param[in] dst_port 远程端端口。
		* @param[in] worldSpace 模型所在的世界坐标系。
		* @param[out] initialPosition 模型初始Tpose下各节点坐标，且按照枚举"_BodyNodes_"来排序。
		* @param[out] initialPosition_rHand 模型初始Tpose下右手各节点坐标，且按照枚举"_HandNodes_"来排序。
		* @param[out] initialPosition_lHand 模型初始Tpose下左手各节点坐标，且按照枚举"_HandNodes_"来排序。
		* @return Ture, 成功；false, 失败。
		* @remark
		*   (1) 枚举"_BodyNodes_"及"_HandNodes_"在头文件"VDMocapSDK_DataReadBH_DataType.h"中定义。
		*   (2) 所述T_pose：模型站立，双腿平行，且双脚脚掌都指向正前方，左右手分别向左右平举，掌心向下。
		*/
		VDMOCAPSDKDATAREAD_API bool UdpGetRecvInitialTposePosition(int index, const char* dst_ip, unsigned short dst_port, _WorldSpace_ worldSpace,
			float initialPosition_body[NODES_BODY][3], float initialPosition_rHand[NODES_HAND][3], float initialPosition_lHand[NODES_HAND][3]);

	}//end namespace

#ifdef __cplusplus
}
#endif

#endif