close all;
clc;
clear;
M = csvread('subject_data_4.csv', 1, 1);

s = 30;
e = length(M);

%%%%% No Filter
% MEAN VALUES
subplot(3, 4, 1)
% figure;

plot(M(s:e,1));
hold on;
plot(M(s:e,2));
plot(M(s:e,3));
title('Mean Values X NF');
legend('x','y','z')

subplot(3, 4,2)
% figure;
% hold on;
plot(M(s:e,4));
hold on;
plot(M(s:e,5));
plot(M(s:e,6));
title('Standard Deviation NF');
legend('x','y','z')

subplot(3, 4,3)
% figure;
% hold on;
plot(M(s:e,7));
hold on;
plot(M(s:e,8));
plot(M(s:e,9));
title('min max NF');
legend('x','y','z')

subplot(3, 4,4)
% figure;
% hold on;
plot(M(s:e,10));
hold on;
plot(M(s:e,11));
plot(M(s:e,12));
title('RMS NF');
legend('x','y','z')


%%%%% LPF Filter
% MEAN VALUES
subplot(3, 4,5)
% figure;
% hold on;
plot(M(s:e,13));
hold on;
plot(M(s:e,14));
plot(M(s:e,15));
title('Mean Values X DC');
legend('x','y','z')

subplot(3, 4,6)
% figure;
% hold on;
plot(M(s:e,16));
hold on;
plot(M(s:e,17));
plot(M(s:e,18));
title('Standard Deviation DC');
legend('x','y','z')

subplot(3, 4,7)
% figure;
% hold on;
plot(M(s:e,19));
hold on;
plot(M(s:e,20));
plot(M(s:e,21));
title('min max DC');
legend('x','y','z')

subplot(3, 4,8)
% figure;
% hold on;
plot(M(s:e,22));
hold on;
plot(M(s:e,23));
plot(M(s:e,24));
title('rms DC');
legend('x','y','z')

%%%%% HPF Filter
% MEAN VALUES
subplot(3, 4,9)
% figure;
hold on;
plot(M(s:e,25));
hold on;
plot(M(s:e,26));
plot(M(s:e,27));
title('Mean Values X AC');
legend('x','y','z')

subplot(3, 4,10)
% figure;
% hold on;
plot(M(s:e,28));
hold on;
plot(M(s:e,29));
plot(M(s:e,30));
title('Standard Deviation AC');
legend('x','y','z')

subplot(3, 4,11)
% figure;
% hold on;
plot(M(s:e,31));
hold on;
plot(M(s:e,32));
plot(M(s:e,33));
title('min max AC');
legend('x','y','z')

subplot(3, 4,12)
% figure;
% hold on;
plot(M(s:e,34));
hold on;
plot(M(s:e,35));
plot(M(s:e,36));
title('rms AC');
legend('x','y','z')

figure;
plot(M(s:e,37));
hold on
plot(M(s:e,38));
plot(M(s:e,39));