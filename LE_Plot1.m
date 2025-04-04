x = xlsread('obser.xlsx');
x1=xlsread('AFCEnK_Lor.xlsx')';
t=0:.05:4.95;
N=100;
for j=1:N
LE=lyapunovExponent(x(:,j));
LE_arr(:,j)=[LE];    
LE1=lyapunovExponent(x1(:,j));
LE1_arr(:,j)=[LE1];
end
%plot RMSE
figure;
hold on;
plot( t, LE_arr', 'b<-','MarkerSize',2,'Linewidth',2 ); 
plot( t, LE1_arr', 'c.-','MarkerSize',2,'Linewidth',2 );
title('Values of estimated Lyapunov exponent for logistic map for r = 0...4.95');
xlabel('Update time step $\Delta x$','Interpreter','latex');
ylabel('Values of estimated Lyapunov exponent');
legend('lyapunov exponent lorenz96','lyapunov exponent-AFCEnKF-ML');




