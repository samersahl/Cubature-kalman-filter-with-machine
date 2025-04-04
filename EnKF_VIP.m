tic
H=eye(40);
F=8;
dt=.05;
N=[100];
R=.1*eye(40);
steps=100;
I=ones(1,100)';
x0_sum=0;
ensemble_sum=0;
number_of_ensemble=100;
x = xlsread('x.xlsx');
x=x';
for j=1:100
 z_in=5*[tanh(x(1,j)+x(40,j));tanh(x(2,j)+x(1,j));tanh(x(3,j)+x(2,j));tanh(x(4,j)+x(3,j));tanh(x(5,j)+x(4,j));tanh(x(6,j)+x(5,j));tanh(x(7,j)+x(6,j));tanh(x(8,j)+x(7,j));tanh(x(9,j)+x(8,j));tanh(x(10,j)+x(9,j));tanh(x(11,j)+x(10,j));tanh(x(12,j)+x(11,j));tanh(x(13,j)+x(12,j));tanh(x(14,j)+x(13,j));tanh(x(15,j)+x(14,j));tanh(x(16,j)+x(15,j));tanh(x(17,j)+x(16,j));tanh(x(18,j)+x(17,j));tanh(x(19,j)+x(18,j));tanh(x(20,j)+x(19,j));tanh(x(21,j)+x(20,j));tanh(x(22,j)+x(21,j));tanh(x(23,j)+x(22,j));tanh(x(24,j)+x(23,j));tanh(x(25,j)+x(24,j));tanh(x(26,j)+x(25,j));tanh(x(27,j)+x(26,j));tanh(x(28,j)+x(27,j));tanh(x(29,j)+x(28,j));tanh(x(30,j)+x(29,j));tanh(x(31,j)+x(30,j));tanh(x(32,j)+x(31,j));tanh(x(33,j)+x(32,j));tanh(x(34,j)+x(33,j));tanh(x(35,j)+x(34,j));tanh(x(36,j)+x(35,j));tanh(x(37,j)+x(36,j));tanh(x(38,j)+x(37,j));tanh(x(39,j)+x(38,j));tanh(x(40,j)+x(39,j))]+sqrtm(.1)*randn(40,1);
 z_in_arr(:,j)=[z_in];
end
for j=1:100
   x0= normrnd(0,1,[40,1]);
   ensemble_arr1(:,j)=[x0];
   x0_sum=x0_sum+x0;
   end  
  x0_bar=x0_sum/100;
  s=(ensemble_arr1- x0_bar*I')*(ensemble_arr1- x0_bar*I')'/sqrt(number_of_ensemble-1);
  for j=1:100
  for j=1:number_of_ensemble 
     ensemble=[(ensemble_arr1(2,j)-ensemble_arr1(38,j))*ensemble_arr1(39,j)-ensemble_arr1(1,j)+F;(ensemble_arr1(3,j)-ensemble_arr1(39,j))*ensemble_arr1(40,j)-ensemble_arr1(2,j)+F;(ensemble_arr1(4,j)-ensemble_arr1(40,j))*ensemble_arr1(1,j)-ensemble_arr1(3,j)+F;(ensemble_arr1(5,j)-ensemble_arr1(2,j))*ensemble_arr1(3,j)-ensemble_arr1(4,j)+F;(ensemble_arr1(6,j)-ensemble_arr1(3,j))*ensemble_arr1(4,j)-ensemble_arr1(5,j)+F;(ensemble_arr1(7,j)-ensemble_arr1(4,j))*ensemble_arr1(5,j)-ensemble_arr1(6,j)+F;(ensemble_arr1(8,j)-ensemble_arr1(5,j))*ensemble_arr1(6,j)-ensemble_arr1(7,j)+F;(ensemble_arr1(9,j)-ensemble_arr1(6,j))*ensemble_arr1(7,j)-ensemble_arr1(8,j)+F;(ensemble_arr1(10,j)-ensemble_arr1(7,j))*ensemble_arr1(8,j)-ensemble_arr1(9,j)+F;(ensemble_arr1(11,j)-ensemble_arr1(8,j))*ensemble_arr1(9,j)-ensemble_arr1(10,j)+F;(ensemble_arr1(12,j)-ensemble_arr1(9,j))*ensemble_arr1(10,j)-ensemble_arr1(11,j)+F;(ensemble_arr1(13,j)-ensemble_arr1(10,j))*ensemble_arr1(11,j)-ensemble_arr1(12,j)+F;(ensemble_arr1(14,j)-ensemble_arr1(11,j))*ensemble_arr1(12,j)-ensemble_arr1(13,j)+F;(ensemble_arr1(15,j)-ensemble_arr1(12,j))*ensemble_arr1(13,j)-ensemble_arr1(14,j)+F;(ensemble_arr1(16,j)-ensemble_arr1(13,j))*ensemble_arr1(14,j)-ensemble_arr1(15,j)+F;(ensemble_arr1(17,j)-ensemble_arr1(14,j))*ensemble_arr1(15,j)-ensemble_arr1(16,j)+F;(ensemble_arr1(18,j)-ensemble_arr1(15,j))*ensemble_arr1(16,j)-ensemble_arr1(17,j)+F;(ensemble_arr1(19,j)-ensemble_arr1(16,j))*ensemble_arr1(17,j)-ensemble_arr1(18,j)+F;(ensemble_arr1(20,j)-ensemble_arr1(17,j))*ensemble_arr1(18,j)-ensemble_arr1(19,j)+F;(ensemble_arr1(21,j)-ensemble_arr1(18,j))*ensemble_arr1(19,j)-ensemble_arr1(20,j)+F;(ensemble_arr1(22,j)-ensemble_arr1(19,j))*ensemble_arr1(20,j)-ensemble_arr1(21,j)+F;(ensemble_arr1(23,j)-ensemble_arr1(20,j))*ensemble_arr1(21,j)-ensemble_arr1(22,j)+F;(ensemble_arr1(24,j)-ensemble_arr1(21,j))*ensemble_arr1(22,j)-ensemble_arr1(23,j)+F;(ensemble_arr1(25,j)-ensemble_arr1(22,j))*ensemble_arr1(23,j)-ensemble_arr1(24,j)+F;(ensemble_arr1(26,j)-ensemble_arr1(23,j))*ensemble_arr1(24,j)-ensemble_arr1(25,j)+F;(ensemble_arr1(27,j)-ensemble_arr1(24,j))*ensemble_arr1(25,j)-ensemble_arr1(26,j)+F;(ensemble_arr1(28,j)-ensemble_arr1(25,j))*ensemble_arr1(26,j)-ensemble_arr1(27,j)+F;(ensemble_arr1(29,j)-ensemble_arr1(26,j))*ensemble_arr1(27,j)-ensemble_arr1(28,j)+F;(ensemble_arr1(30,j)-ensemble_arr1(27,j))*ensemble_arr1(28,j)-ensemble_arr1(29,j)+F;(ensemble_arr1(31,j)-ensemble_arr1(28,j))*ensemble_arr1(29,j)-ensemble_arr1(30,j)+F;(ensemble_arr1(32,j)-ensemble_arr1(29,j))*ensemble_arr1(30,j)-ensemble_arr1(31,j)+F;(ensemble_arr1(33,j)-ensemble_arr1(30,j))*ensemble_arr1(31,j)-ensemble_arr1(32,j)+F;(ensemble_arr1(34,j)-ensemble_arr1(31,j))*ensemble_arr1(32,j)-ensemble_arr1(33,j)+F;(ensemble_arr1(35,j)-ensemble_arr1(32,j))*ensemble_arr1(33,j)-ensemble_arr1(34,j)+F;(ensemble_arr1(36,j)-ensemble_arr1(33,j))*ensemble_arr1(34,j)-ensemble_arr1(35,j)+F;(ensemble_arr1(37,j)-ensemble_arr1(34,j))*ensemble_arr1(35,j)-ensemble_arr1(36,j)+F;(ensemble_arr1(38,j)-ensemble_arr1(35,j))*ensemble_arr1(36,j)-ensemble_arr1(37,j)+F;(ensemble_arr1(39,j)-ensemble_arr1(36,j))*ensemble_arr1(37,j)-ensemble_arr1(38,j)+F;(ensemble_arr1(40,j)-ensemble_arr1(37,j))*ensemble_arr1(38,j)-ensemble_arr1(39,j)+F;(ensemble_arr1(1,j)-ensemble_arr1(38,j))*ensemble_arr1(39,j)-ensemble_arr1(40,j)+F]+sqrtm(.1)*randn(40,1);
     ensemble_arr1(:,j)=[ensemble];
     ensemble_sum=ensemble_sum+ensemble;
  end
   ensemble_bar=ensemble_sum/number_of_ensemble;
   Px=(ensemble_arr1- ensemble_bar*I')*(ensemble_arr1- ensemble_bar*I')'/(number_of_ensemble-1);
   zhat_sum=0;
   for j=1:number_of_ensemble
        zhat=H*ensemble_arr1(:,j)+sqrtm(R)*randn(40,1);
        zhat_arr(:,j)=[zhat];
        zhat_sum=zhat_sum+zhat;
   end
   zhat_bar=zhat_sum/number_of_ensemble;
   Pz=(zhat_arr- zhat_bar*I')*(zhat_arr- zhat_bar*I')'/(number_of_ensemble-1)+R;
   Pxz=(ensemble_arr1- ensemble_bar*I')*(zhat_arr- zhat_bar*I')'/(number_of_ensemble-1);
   K_k=Pxz*inv(Pz);
   ensemble_sum=0;
    for j=1:number_of_ensemble
     x_k=ensemble_arr1(:,j)+K_k*(z_in_arr(:,j)-zhat_arr(:,j));
     ensemble_arr1(:,j)=[x_k];
     ensemble_sum=ensemble_sum+x_k;
    end
    ensemble_bar=ensemble_sum/number_of_ensemble;
    P=(ensemble- ensemble_bar*I')*(ensemble- ensemble_bar*I')'/(number_of_ensemble-1); 
    P_k1=diag(diag(P));
   Normalized_data1=normalize(ensemble_arr1);
  end
  toc