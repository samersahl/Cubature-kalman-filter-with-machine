x = xlsread('obser1.xlsx');
x1=xlsread('AFCEnK_R.xlsx');
t=1:1:4;
n=4;
for j=1:n
LE=lyapunovExponent(x(:,j));
LE_arr(:,j)=[LE];    
LE1=lyapunovExponent(x1(:,j));
LE1_arr(:,j)=[LE1];
end
%plot RMSE
figure;
hold on;
plot( t, LE_arr', 'b<-','MarkerSize',2,'Linewidth',2 );
plot( t, LE1_arr', 'k+-','MarkerSize',2,'Linewidth',2 );
title('Values of estimated Lyapunov exponent for logistic map for n=4');
xlabel('Lyapunov exponent n');
ylabel('Values of estimated Lyapunov exponent');
legend('lyapunov exponent HR','lyapunov exponent-AFCEnKF-ML');





