clc;
clear;
close all;
fc = 16;   % Cutoff frequency
fs = 52;  % Sampling frequency
fn = fs/2;  % Nyquist frequency

% Butterworth Filter
[b, a] = butter(50, fc/fn);
freqz(b,a)
% 
% % Chebyshev Type 1
% figure;
% [b, a] = cheby1(5, 10, fc/fn);
% freqz(b,a)

% moving average filter
% figure;
% N = 2;
% a = 1;
% b = ones(1,N);
% b = 3* b/N;
% [h, w] = freqz(b, a, 'whole', 2001);
% plot(w/pi,20*log10(abs(h)))
% ax = gca;
% ax.YLim = [-50 5];
% ax.XLim = [0 1];
% ax.XTick = 0:.1:1;
% xlabel('Normalized Frequency (\times\pi rad/sample)')
% ylabel('Magnitude (dB)')
% 
% figure;
% plot(w/pi,abs(h))
% ax = gca;
% ax.YLim = [0 3];
% ax.XLim = [0 1];
% ax.XTick = 0:.1:1;
% xlabel('Normalized Frequency (\times\pi rad/sample)')
% ylabel('Magnitude (dB)')

