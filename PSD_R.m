x = xlsread('obser1.xlsx');
x1=xlsread('AFCEnK_R.xlsx');
segmentLength = 64;
overlap = 32;
segmentLength1 = 64;
overlap1 = 32;
[psd, freq] = pwelch(x, segmentLength, overlap);
[psd1]=pwelch(x1, segmentLength1, overlap1);
psd_avg = mean(psd, 2);
psd_avg1 = mean(psd1, 2);
%plot RMSE
figure;
hold on;
plot( freq, 10*log10(psd_avg), 'k<-','MarkerSize',2,'Linewidth',2 ); 
plot( freq, 10*log10(psd_avg1), 'r.-','MarkerSize',2,'Linewidth',2 );
xlabel('Frequency (Hz)');
ylabel('Power density spectrum');
legend('PSD-R','PSD-AFCEnKF-ML');
