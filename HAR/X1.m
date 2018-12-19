clc;
clear;
close all;
M = csvread('subject1.csv', 0, 1);
N = 5000;
x = M(:,1);
y = M(:,2);
z = M(:,3);
l = M(:,4);
plot(x);
hold on;
plot(y);
plot(z);
ylabel('Acceleration')
yyaxis right
plot(l, 'LineWidth', 2);
ylabel('Activity');
title('Subject 1`');
xlabel('Time Index');
legend('x', 'y', 'z', 'label');

figure;
subplot(2,1,1)
plot(x);
title('Subject 1 X-Acceration`');
ylabel('Acceleration')
xlabel('Time Index');
subplot(2,1,2)
plot(x(50000:51000))
title('Subject 1 X-Acceration Walking`');
ylabel('Acceleration')
xlabel('Time Index (+50000)');
% 
% M = csvread('subject3.csv', 0, 1);
% N = 5000;
% figure;
% plot(M(1:N,1), '.');
% hold on;
% plot(M(1:N,2), '.');
% plot(M(1:N,3), '.');
% title('Subject 3');
% 
% 
% M = csvread('subject5.csv', 0, 1);
% N = 5000;
% figure;
% plot(M(1:N,1), '.');
% hold on;
% plot(M(1:N,2), '.');
% plot(M(1:N,3), '.');
% title('Subject 5');
% 
% M = csvread('subject6.csv', 0, 1);
% N = 5000;
% figure;
% plot(M(1:N,1), '.');
% hold on;
% plot(M(1:N,2), '.');
% plot(M(1:N,3), '.');
% title('Subject 6');
% 
% M = csvread('subject10.csv', 0, 1);
% N = 5000;
% figure;
% plot(M(1:N,1), '.');
% hold on;
% plot(M(1:N,2), '.');
% plot(M(1:N,3), '.');
% title('Subject 10');
% 
% M = csvread('subject13.csv', 0, 1);
% N = 5000;
% figure;
% plot(M(1:N,1), '.');
% hold on;
% plot(M(1:N,2), '.');
% plot(M(1:N,3), '.');
% title('Subject 13');