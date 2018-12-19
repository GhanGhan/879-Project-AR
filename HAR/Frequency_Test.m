clc;
clear;
close all;
%Constants
M = csvread('subject1.csv', 0, 1);
L = 156;     % Length of window
S = 110000;  % Starting index
N = S+L;   % Last index 
fc = 5;   % Cutoff frequency
fs = 52;  % Sampling frequency
fn = fs/2;  % Nyquist frequency
%////////// SUBJECT 

% Raw Values
x = M(S:N,1);
y = M(S:N,2);
z = M(S:N,3);
maxV = max([max(x) max(y) max(z)]);
% Normalized Values
xn = x/maxV;
yn = y/maxV;
zn = z/maxV;
plot(zn)

X = fft(zn);
P2 = abs(X/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:(L/2))/L;
figure;

len = length(P1);
fAC = f(2:len);
P1AC = P1(2:len);

plot(fAC,P1AC,'.') %Remove DC Component from plot
title('Single-Sided Amplitude Spectrum of X(t) (Normalized)')
xlabel('f (Hz)')
ylabel('|P1(f)|')
fprintf('Normalized mean value: %0.4f \n', mean(zn))
fprintf('Normalized standard dev: %0.4f \n', std(zn))
fprintf('Raw mean value: %0.4f \n', mean(z))
fprintf('Raw standard dev: %0.4f \n', std(z))
[v, i ] = max(P1AC);
i = fAC(i);
fprintf('Largest AC Frequency: %d \n', i); 
fprintf('Has magnitude: %0.5f \n', v);