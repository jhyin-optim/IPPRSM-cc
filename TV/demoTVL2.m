%
% Demo TV/L2 solve
%
% Suppose the data accuquisition model is given by: d = K*Wbar ,
% where Wbar is an original image, K is a convolution matrix 
% and d is a blurry and noisy observation. To recover
% Wbar from d and K, we solve TV/L2 (ROF) model
%
% ***     min_W \sum_i | |Di*W||_{1} + delta/2*||K*W - d||^2      ***
%

clear; close all;
% path(path,genpath(pwd));
addpath('./Images')
rng(2016);

%% generate data -- see nested function below
Ima = {'cameraman.tif','barbara.png','lena.png','chart.tiff','circles.tif', ... 
            'housergb.png','shape.jpg','TwoCircles.tif'};
index = 7;
I = double(imread(Ima{index}))/255;   
        
[m,n] = size(I);
% H = fspecial('disk',7);
H = fspecial('average',10);  % 此时Ima{6}效果不好，12对应效果好 % 10
% H = fspecial('average',15); %创建一个15×15的均值滤波器 
         
gamma = 2.e-3;     
Bn = imfilter(I,H,'circular','conv') + gamma*randn(m,n);
delta = 5.e3; %正则参数大于0

snrBn = snr(Bn,I);  % The initial value of SNR 
fprintf(' ********** Comparison starts **********\n');

%% Run SCPRSM
 fprintf(' ********** run SCPRSM ***************\n');
 t1 = cputime;
 out1 = SCPRSM(H,Bn,delta); 
 t1 = cputime - t1;
 
%% Run GrpADMM
fprintf(' ********** run GrpADMM ***************\n');
t2 = cputime;
out2 = GrpADMM(H,Bn,delta); 
t2 = cputime - t2;

%% Run IPPRSM_cc 
fprintf(' ********** run IPPRSM_cc ***************\n');
t3 = cputime;
out3 = IPPRSM_cc(H,Bn,delta); 
t3 = cputime - t3;

% output results
fid=fopen('mytext.txt','w');
fprintf(fid,'%s & %.2f & %d/%.2f/%.2f & %d/%.2f/%.2f & %d/%.2f/%.2f\\\\\n', ... 
   Ima{index},snrBn,out1.itr,t1,snr(out1.sol),out2.itr,t2,snr(out2.sol),out3.itr,t3,snr(out3.sol));
% mean(progress_r)
fclose(fid);
fprintf('SCPRSM SNR(Bn) %4.2fdB, SNR(Recovered) %4.2fdB,',snrBn,snr(out1.sol));
fprintf(' CPU %4.2fs, Iteration %d\n\n',t1,out1.itr);

fprintf('GrpADMM SNR(Bn) %4.2fdB, SNR(Recovered) %4.2fdB,',snrBn,snr(out2.sol));
fprintf(' CPU %4.2fs, Iteration %d\n\n',t2,out2.itr);

fprintf('IPPRSM_cc SNR(Bn) %4.2fdB, SNR(Recovered) %4.2fdB,',snrBn,snr(out3.sol));
fprintf(' CPU %4.2fs, Iteration %d\n\n',t3,out3.itr);

%% plot result
figure(1);
imshow(I,[]);
title('Original image');

figure(2);
imshow(Bn,[]);
title('Degraded image');

figure(3);
imshow(out1.sol,[]);
title('SCPRSM');

figure(4);
imshow(out2.sol,[]);
title('GrpADMM');

figure(5);
imshow(out3.sol,[]);
title('IPPRSM-cc ');

% fig = figure(6);
% plot(1:length(out1.snr),out1.snr,'-xb',1:length(out2.snr),out2.snr,'-oy',1:length(out3.snr),out3.snr,'-*r');
% %title('SNR (dB)','fontsize',13);
% legend('SCPRSM','GrpADMM','IPPRSM-cc');%,
% ylabel('SNR (dB) ', 'fontsize', 14, 'interpreter', 'latex');%'$(\Phi(x^k) - \Phi^*)/\Phi^*$',
% xlabel('iteration');
% print(fig, '-depsc','admm.eps');
% 
% fig = figure(7);
% semilogy(out1.time, out1.snr, '-xb', 'LineWidth',2.5);
% hold on
% semilogy(out2.time, out2.snr, '-oy', 'LineWidth',2.5);
% hold on
% semilogy(out3.time, out3.snr, '-*r', 'LineWidth',2.5);
% legend('SCPRSM','GrpADMM','IPPRSM-cc')
% ylabel('SNR (dB) ','fontsize', 14, 'interpreter', 'latex');%$(\Phi(x^k) - \Phi^*)/\Phi^*$',
% xlabel('CPU time (in seconds)');
% print(fig, '-depsc','admm.eps');


% figure(1);
% subplot(121); imshow(Bn,[]);
% title(sprintf('SNR %4.2fdB',snrBn),'fontsize',13);
% subplot(122); imshow(out1.sol,[]);
% title(sprintf('SNR %4.2fdB, CPU %4.2fs, It: %d',snr(out1.sol),t1,out1.itr),'fontsize',13);
% fig = figure(8);
% semilogy(1:length(out1.f)-1,out1.f(2:end),'-xb',1:length(out3.f)-1,out3.f(2:end),'-*r');
% %title('Function values','fontsize',13);
% legend('SCPRSM','GrpADMM','IPPRSM-cc');
% ylabel('Function values ', 'fontsize', 14, 'interpreter', 'latex');%'$(\Phi(x^k) - \Phi^*)/\Phi^*$',
% xlabel('iteration');
% print(fig, '-depsc','admm.eps');
%axis([0,inf,min(out1.f(2:end)),max(out3.f(2:end)),max(out4.f(2:end))]);


