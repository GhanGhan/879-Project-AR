clc;
clear;
close all;
%Constants
M = csvread('subject1.csv', 0, 1);
L = 128;     % Length of window
S = 70000;  % Starting index
N = S+L;   % Last index 
fc = 18;   % Cutoff frequency
fs = 52;  % Sampling frequency
fn = fs/2;  % Nyquist frequency
Order = 3;  %Order of Filter
Rp = 0.01;   %Ripple
Rs = 40;
%////////// SUBJECT 

% Raw Values ///////////////////////////////////////
x = M(S:N,1);
y = M(S:N,2);
z = M(S:N,3);

% Normalized Values//////////////////////////////////
maxV = max([max(x) max(y) max(z)]);
xn = x/maxV;
yn = y/maxV;
zn = z/maxV;
% Raw Plot
figure;
plot(xn, '.');
hold on;
plot(yn, '.');
plot(zn, '.');
title('Subject (Normalized RAW)');
legend('x', 'y', 'z');

% Moving Average filter///////////////////////////////
xa = movmean(x, 3);
ya = movmean(y, 3); 
za = movmean(z, 3); 

maxVa = max([max(xa) max(ya) max(za)]);
xan = xa/maxVa;
yan = ya/maxVa;
zan = za/maxVa;
% Moving Average Plot
figure;
plot(xan, '.');
hold on;
plot(yan, '.');
plot(zan, '.');
title('Subject Moving Average)');
legend('x', 'y', 'z');

% Median Filter/////////////////////////////////////////
xm = medfilt1(x);
ym = medfilt1(y);
zm = medfilt1(z);

maxVm = max([max(xm) max(ym) max(zm)]);
xmn = x/maxVm;
ymn = y/maxVm;
zmn = z/maxVm;
% Median Plot
figure;
plot(xmn, '.');
hold on;
plot(ymn, '.');
plot(zmn, '.');
title('Subject Median)');
legend('x', 'y', 'z');



% Frequency Shit %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Normalized /////////////////////////////
X = fft(zn);
P2 = abs(X/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:(L/2))/L;
figure;

len = length(P1);
fAC = f(2:len);
P1AC = P1(2:len);

stem(fAC,P1AC) 
title('Single-Sided Amplitude Spectrum of X(t) (Normalized)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

% Moving Average/////////////////////////////
X = fft(zan);
P2 = abs(X/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:(L/2))/L;
figure;

len = length(P1);
fAC = f(2:len);
P1AC = P1(2:len);

stem(fAC,P1AC) 
title('Single-Sided Amplitude Spectrum of X(t) (Moving Average Normalized)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

% Median /////////////////////////////
X = fft(zmn);
P2 = abs(X/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:(L/2))/L;
figure;

len = length(P1);
fAC = f(2:len);
P1AC = P1(2:len);

stem(fAC,P1AC) 
title('Single-Sided Amplitude Spectrum of X(t) (Median Normalized)')
xlabel('f (Hz)')
ylabel('|P1(f)|')
