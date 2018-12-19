clc;
clear;
close all;
%////////// SUBJECT 2
L = 52;
S = 90000;
M = csvread('subject2.csv', 0, 1);
N = S+L;
% Raw Values
x = M(S:N,1);
y = M(S:N,2);
z = M(S:N,3);
maxV = max([max(x) max(y) max(z)]);
% Normalized Values
xn = x/maxV;
yn = y/maxV;
zn = z/maxV;
% Raw Plot
figure;
plot(x, '.');
hold on;
plot(y, '.');
plot(z, '.');
title('Subject 2 (RAW)');
legend('x', 'y', 'z');
% Normalized Plot
figure;
plot(xn, '.');
hold on;
plot(yn, '.');
plot(zn, '.');
title('Subject 2 (Normalized)');
legend('xn', 'yn', 'zn');

%Moving Average Filter
windowSize = 10;
b = (1/windowSize)*ones(1, windowSize);
a = 1;
% Normalized and Filtered Values
xnf = filter(b,a, xn);
ynf = filter(b,a, yn);
znf = filter(b,a, zn);

%Normalized and Filtered Plot
figure;
plot(xnf, '.');
hold on;
plot(ynf, '.');
plot(znf, '.');
title('Subject 2 (Normalized and MA Filtered)');
legend('xn', 'yn', 'zn');

% Filtered Values
xf = filter(b,a, x);
yf = filter(b,a, y);
zf = filter(b,a, z);

% Filtered Plot
figure;
plot(xf, '.');
hold on;
plot(yf, '.');
plot(zf, '.');
title('Subject 2 (MA Filtered)');
legend('x', 'y', 'z');

%Filtered and Normalized values
maxV = max([max(xf) max(yf) max(zf)]);
xfn = xf/maxV;
yfn = yf/maxV;
zfn = zf/maxV;

% Filtered and Normalized Plot
figure;
plot(xfn, '.');
hold on;
plot(yfn, '.');
plot(zfn, '.');
title('Subject 2 (MA and Normalized)');
legend('x', 'y', 'z');

% Frequency Domain
X = fft(X);
P2 = a