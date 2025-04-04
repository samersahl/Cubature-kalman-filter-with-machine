H=eye(4);
dt=.01;
N=[200];
R=.01*eye(4);
E=eye(4);
Q=.01*eye(4);
m=8;
a = 0.25;
b = 3;
c = 0.5;
d = 0.05;
kesi1=sqrt(N)*eye(4);                    %the 𝑖-th point of [1]
kesi2=-sqrt(N)*eye(4);
kes1=kesi1(1,:)';
kes2=kesi1(2,:)';
kes3=kesi1(3,:)';
kes4=kesi1(4,:)';
kes5=kesi2(1,:)';
kes6=kesi2(2,:)';
kes7=kesi2(3,:)';
kes8=kesi2(4,:)';
steps=200;
I=ones(1,200)';
x = xlsread('obser1.xlsx');
x=x';
for j=1:200
 z_in_arr=H*x+sqrtm(.01)*randn(4,200);
end

 x0_sum=0;
number_of_ensemble=200;
for j=1:200
   x0= normrnd(0,1,[4,1]);
   ensemble_arr(:,j)=[x0];
   x0_sum=x0_sum+x0;
end 
  x0_bar=x0_sum/200;
  s=(ensemble_arr- x0_bar*I')*1/sqrt(number_of_ensemble-1);
  [U,S,V] =svd(s,0);
  SS=S(1:4,1:4);
  Spost=U*SS*U';
  for j=1:number_of_ensemble 
  for j=1:number_of_ensemble 
      ensemble=[-ensemble_arr(2,j)-ensemble_arr(3,j);ensemble_arr(1,j)+a.*ensemble_arr(2,j)+ensemble_arr(4,j);b+ensemble_arr(1,j).*ensemble_arr(3,j);-c.*ensemble_arr(3,j)+d.*ensemble_arr(4,j)];
      ensemble_arr(:,j)=[ensemble];
      %Calculation of Cubature ensemble Points
  xminus_sum=0;
   for j=1:number_of_ensemble
    rjpoint1=Spost*kes1+ensemble_arr(:,j);          %The cubature points 
    rjpoint1_arr(:,j)=[rjpoint1];
  rjpoint2=Spost*kes2+ensemble_arr(:,j);
  rjpoint2_arr(:,j)=[rjpoint2];
    rjpoint3=Spost*kes3+ensemble_arr(:,j);
    rjpoint3_arr(:,j)=[rjpoint3];
    rjpoint4=Spost*kes4+ensemble_arr(:,j);
    rjpoint4_arr(:,j)=[rjpoint4];
    rjpoint5=Spost*kes5+ensemble_arr(:,j);
    rjpoint5_arr(:,j)=[rjpoint5];
    rjpoint6=Spost*kes6+ensemble_arr(:,j);
    rjpoint6_arr(:,j)=[rjpoint6];
    rjpoint7=Spost*kes7+ensemble_arr(:,j);
    rjpoint7_arr(:,j)=[rjpoint7];
    rjpoint8=Spost*kes8+ensemble_arr(:,j);
    rjpoint8_arr(:,j)=[rjpoint8];
    Xminus1=[-rjpoint1_arr(2,j)-rjpoint1_arr(3,j);rjpoint1_arr(1,j)+a.*rjpoint1_arr(2,j)+rjpoint1_arr(4,j);b+rjpoint1_arr(1,j).*rjpoint1_arr(3,j);-c.*rjpoint1_arr(3,j)+d.*rjpoint1_arr(4,j)];
    rjpoint1_arr=rjpoint2_arr;
    Xminus2=[-rjpoint1_arr(2,j)-rjpoint1_arr(3,j);rjpoint1_arr(1,j)+a.*rjpoint1_arr(2,j)+rjpoint1_arr(4,j);b+rjpoint1_arr(1,j).*rjpoint1_arr(3,j);-c.*rjpoint1_arr(3,j)+d.*rjpoint1_arr(4,j)];
    rjpoint1_arr=rjpoint3_arr;
    Xminus3=[-rjpoint1_arr(2,j)-rjpoint1_arr(3,j);rjpoint1_arr(1,j)+a.*rjpoint1_arr(2,j)+rjpoint1_arr(4,j);b+rjpoint1_arr(1,j).*rjpoint1_arr(3,j);-c.*rjpoint1_arr(3,j)+d.*rjpoint1_arr(4,j)];
    rjpoint1_arr=rjpoint4_arr;
    Xminus4=[-rjpoint1_arr(2,j)-rjpoint1_arr(3,j);rjpoint1_arr(1,j)+a.*rjpoint1_arr(2,j)+rjpoint1_arr(4,j);b+rjpoint1_arr(1,j).*rjpoint1_arr(3,j);-c.*rjpoint1_arr(3,j)+d.*rjpoint1_arr(4,j)];
    rjpoint1_arr=rjpoint5_arr;
    Xminus5=[-rjpoint1_arr(2,j)-rjpoint1_arr(3,j);rjpoint1_arr(1,j)+a.*rjpoint1_arr(2,j)+rjpoint1_arr(4,j);b+rjpoint1_arr(1,j).*rjpoint1_arr(3,j);-c.*rjpoint1_arr(3,j)+d.*rjpoint1_arr(4,j)];
    rjpoint1_arr=rjpoint6_arr;
    Xminus6=[-rjpoint1_arr(2,j)-rjpoint1_arr(3,j);rjpoint1_arr(1,j)+a.*rjpoint1_arr(2,j)+rjpoint1_arr(4,j);b+rjpoint1_arr(1,j).*rjpoint1_arr(3,j);-c.*rjpoint1_arr(3,j)+d.*rjpoint1_arr(4,j)];
    rjpoint1_arr=rjpoint7_arr;
    Xminus7=[-rjpoint1_arr(2,j)-rjpoint1_arr(3,j);rjpoint1_arr(1,j)+a.*rjpoint1_arr(2,j)+rjpoint1_arr(4,j);b+rjpoint1_arr(1,j).*rjpoint1_arr(3,j);-c.*rjpoint1_arr(3,j)+d.*rjpoint1_arr(4,j)];
    rjpoint1_arr=rjpoint8_arr;
    Xminus8=[-rjpoint1_arr(2,j)-rjpoint1_arr(3,j);rjpoint1_arr(1,j)+a.*rjpoint1_arr(2,j)+rjpoint1_arr(4,j);b+rjpoint1_arr(1,j).*rjpoint1_arr(3,j);-c.*rjpoint1_arr(3,j)+d.*rjpoint1_arr(4,j)];
    xminus=(1/m)*(Xminus1+Xminus2+Xminus3+Xminus4+Xminus5+Xminus6+Xminus7+Xminus8)+diag(diag(randn(4,1))*sqrtm(.1));
    xminus_arr(:,j)=[xminus];
    xminus_sum=xminus_sum+xminus;
   end  
  xminus_bar=xminus_sum/number_of_ensemble;
  s1=(xminus_arr- xminus_bar*I')*1/sqrt(number_of_ensemble-1);
  PP=s1*s1'+Q;
  [U,S,V] =svd(s1,0);
  SS1=S(1:4,1:4);
  Spost1=U*SS*U';
  %Calculation of Cubature ensemble Points measurment
  zminus_sum=0;

    for j=1:number_of_ensemble
    zjpoint1=Spost1*kes1+xminus_arr(:,j);          %The cubature points 
    zjpoint2=Spost1*kes2+xminus_arr(:,j);
    zjpoint3=Spost1*kes3+xminus_arr(:,j);
    zjpoint4=Spost1*kes4+xminus_arr(:,j);
    zjpoint5=Spost1*kes5+xminus_arr(:,j);
    zjpoint6=Spost1*kes6+xminus_arr(:,j);
    zjpoint7=Spost1*kes7+xminus_arr(:,j);
    zjpoint8=Spost1*kes8+xminus_arr(:,j);
   
  Zminus1=H*zjpoint1;          
    Zminus2=H*zjpoint2;
    Zminus3=H*zjpoint3;          
    Zminus4=H*zjpoint4;
    Zminus5=H*zjpoint5;          
    Zminus6=H*zjpoint6;
    Zminus7=H*zjpoint7;          
    Zminus8=H*zjpoint8;        
    zminus=(1/m)*(Zminus1+Zminus2+Zminus3+Zminus4+Zminus5+Zminus6+Zminus7+Zminus8)+diag(diag(randn(4,1))*sqrtm(.01));
    zminus_arr(:,j)=[zminus];
    zminus_sum=zminus_sum+zminus;
    
    end

   zminus_bar=zminus_sum/number_of_ensemble;
  s2=(zminus_arr- zminus_bar*I')*1/sqrt(number_of_ensemble-1);
  [U,S,V] =svd(s2,0);
  SS1=S(1:4,1:4);
  Spost2=U*SS*U';
  end
 Pz=(1/(number_of_ensemble-1))*(zminus_arr- zminus_bar*I')*(zminus_arr- zminus_bar*I')'+R;
 EE=(z_in_arr(:,j)-zminus_arr(:,j))*(z_in_arr(:,j)-zminus_arr(:,j))';
 if j>1
   SS=(.95*SS+ EE)/(1+.95);
 else
 SS=E;
 end
 N_=SS-R;
 M_=Pz-R;
 d=trace(N)/trace(Pz);
 if d>=1
     d=d;
 else
     d=1;
 end
 Pxz=d*(1/(number_of_ensemble-1))*(xminus_arr- xminus_bar*I')*(zminus_arr- zminus_bar*I')'; 
 K_k=Pxz*inv(Pz);
  ensemble_sum=0;
  
    for j=1:number_of_ensemble
 x_k=xminus_arr(:,j)+K_k*(z_in_arr(:,j)-zminus_arr(:,j));
 ensemble_arr(:,j)=[x_k];
 ensemble_sum=ensemble_sum+x_k;
  end
    ensemble_bar=ensemble_sum/number_of_ensemble;
    s=(ensemble- ensemble_bar*I')*1/sqrt(number_of_ensemble-1);
    P=s*s';
    P_k=diag(diag(P));
  [U,S,V] =svd(s,0);
  SS=S(1:4,1:4);
  Spost=U*SS*U';
 Normalized_data=normalize(ensemble_arr);

  end

   
  
 
  