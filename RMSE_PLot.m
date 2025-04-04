x = xlsread('obser1.xlsx')';
x1=xlsread('state2.xlsx')';
x2=xlsread('state3.xlsx')';
x3=xlsread('AFCEnK_R.xlsx')';
x4=xlsread('EnKF_R.xlsx')';
t=0:.01:1.99;
N=200;
for j=1:N
RMSE1=sqrt(1/4*(x1(:,j)-x(:,j))'*(x1(:,j)-x(:,j)));
RMSE1_arr(:,j)=[RMSE1];
MRMSE1=mean(RMSE1_arr)
RMSE2=sqrt(1/4*(x2(:,j)-x(:,j))'*(x2(:,j)-x(:,j)));
MRMSE2=mean(RMSE2_arr)
RMSE2_arr(:,j)=[RMSE2];
RMSE3=sqrt(1/4*(x3(:,j)-x(:,j))'*(x3(:,j)-x(:,j)));
RMSE3_arr(:,j)=[RMSE3];
MRMSE3=mean(RMSE3_arr)
RMSE4=sqrt(1/4*(x4(:,j)-x(:,j))'*(x4(:,j)-x(:,j)));
RMSE4_arr(:,j)=[RMSE4];
MRMSE4=mean(RMSE4_arr)
end
%plot RMSE
figure;
hold on;
plot( t, RMSE1_arr', 'k<-','MarkerSize',2,'Linewidth',2 ); 
plot( t, RMSE2_arr', 'r.-','MarkerSize',2,'Linewidth',2 );
plot( t, RMSE3_arr', 'c.-','MarkerSize',2,'Linewidth',2 );
plot( t, RMSE4_arr', 'b.-','MarkerSize',2,'Linewidth',2 );
title(' RMSE ' );
xlabel( 'Update time step $\Delta x$','Interpreter','latex' );
ylabel( 'Estimation RMSE' );
legend('RMSE-AFCEnKF','RMSE-EnKF','RMSE-AFCEnKF-ML','RMSE-EnKF-ML');