clc;
clear;
close all;
%N = 10000;
%////////// SUBJECT 1
% M = csvread('subject1.csv', 0, 1);
% x = M(:,1);
% y = M(:,2);
% z = M(:,3);
% xm = x/max(x);
% ym = y/max(y);
% zm = z/max(z);
% figure;
% plot(x, '.');
% hold on;
% plot(y, '.');
% plot(z, '.');
% title('Subject 1');
% legend('x', 'y', 'z');
% 
% figure;
% plot(xm, '.');
% hold on;
% plot(ym, '.');
% plot(zm, '.');
% title('Subject 1 (Normalized)');
% legend('xm', 'ym', 'zm');

%////////// SUBJECT 2
S = 90000;
M = csvread('subject2.csv', 0, 1);
N = S+52;
x = M(S:N,1);
y = M(S:N,2);
z = M(S:N,3);
xm = x/max(x);
ym = y/max(y);
zm = z/max(z);
figure;
plot(x, '.');
hold on;
plot(y, '.');
plot(z, '.');
title('Subject 2');
legend('x', 'y', 'z');

figure;
plot(xm, '.');
hold on;
plot(ym, '.');
plot(zm, '.');
title('Subject 2 (Normalized)');
legend('xm', 'ym', 'zm');

windowSize = 10;
b = (1/windowSize)*ones(1, windowSize);
a = 1;
xn = filter(b,a, xm);
yn = filter(b,a, ym);
zn = filter(b,a, zm);
figure;
plot(xn, '.');
hold on;
plot(yn, '.');
plot(zn, '.');
title('Subject 2 (Normalized and MA Filtered)');
legend('xn', 'yn', 'zn');

xn = filter(b,a, x);
yn = filter(b,a, y);
zn = filter(b,a, z);
figure;
plot(xn, '.');
hold on;
plot(yn, '.');
plot(zn, '.');
title('Subject 2 (MA Filtered)');
legend('x', 'y', 'z');