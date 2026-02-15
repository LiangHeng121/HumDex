% 加载csv,该CSV只导出了双手的姿态
dataTable = readtable('纯手势动作录制_202617113630.csv', 'ReadVariableNames', true); 
RightHandQuat=table2array(dataTable(:,2:81));
LeftHandQuat=table2array(dataTable(:,83:162));
% 测试左手
idx=520;
QuatLHand=reshape(LeftHandQuat(idx,:),'',20)'; %将数据重组成20*4
[pos,pos_end]=Quat2PositionLHand(QuatLHand);
% 测试右手
idx=26;
QuatRHand=reshape(RightHandQuat(idx,:),'',20)'; %将数据重组成20*4
[pos,pos_end]=Quat2PositionRHand(QuatRHand);










