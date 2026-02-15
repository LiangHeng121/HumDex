 %% 旋转矩阵,转回去的(已核对)
function T=T_SY(q) %转回去的(已核对)
qw=q(1);
qx=q(2);
qy=q(3);
qz=q(4);
T=[ qw^2 + qx^2 - qy^2 - qz^2,   2*qx*qy - 2*qw*qz,   2*qw*qy + 2*qx*qz;...
   2*qw*qz + 2*qx*qy, qw^2 - qx^2 + qy^2 - qz^2,   2*qy*qz - 2*qw*qx;...
   2*qx*qz - 2*qw*qy,   2*qw*qx + 2*qy*qz, qw^2 - qx^2 - qy^2 + qz^2];

