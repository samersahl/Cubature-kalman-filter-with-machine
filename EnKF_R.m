H=eye(4);
dt=.01;
N=[200];
a = 0.25;
b = 3;
c = 0.5;
d = 0.05;
R=.1*eye(4);
steps=200;
I=ones(1,200)';
x0_sum=0;
ensemble_sum=0;
number_of_ensemble=200;
x = xlsread('obser1.xlsx');
x=x';
z_in_arr=H*x+R*ones(4,200);
for j=1:200
   x0= normrnd(0,1,[4,1]);
   ensemble_arr1(:,j)=[x0];
   x0_sum=x0_sum+x0;
   end  
  x0_bar=x0_sum/200;
  s=(ensemble_arr1- x0_bar*I')*(ensemble_arr1- x0_bar*I')'/sqrt(number_of_ensemble-1);
  for j=1:200
  for j=1:number_of_ensemble 
      ensemble=[-ensemble_arr1(2,j)-ensemble_arr1(3,j);ensemble_arr1(1,j)+a.*ensemble_arr1(2,j)+ensemble_arr1(4,j);b+ensemble_arr1(1,j).*ensemble_arr1(3,j);-c.*ensemble_arr1(3,j)+d.*ensemble_arr1(4,j)];
     ensemble_arr1(:,j)=[ensemble];
     ensemble_sum=ensemble_sum+ensemble;
  end
   ensemble_bar=ensemble_sum/number_of_ensemble;
   Px=(ensemble_arr1- ensemble_bar*I')*(ensemble_arr1- ensemble_bar*I')'/(number_of_ensemble-1);
   zhat_sum=0;
   for j=1:number_of_ensemble
        zhat=H*ensemble_arr1(:,j)+sqrtm(R)*randn(4,1);
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