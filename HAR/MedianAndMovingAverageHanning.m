clc;
clear;
close all;
%Constants
M = csvread('subject1.csv', 0, 1);
L = 64;     % Length of window
S = 50000;  % Starting index
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

figure;
plot(x, '.');
hold on;
plot(y, '.');
plot(z, '.');
title('Subject Raw)');
legend('x', 'y', 'z');

% Median Filter/////////////////////////////////////////
% Apply Median Filter
xm = medfilt1(x);
ym = medfilt1(y);
zm = medfilt1(z);

% Normalize the Data
maxVm = max([max(xm) max(ym) max(zm)]);
xmn = x/maxVm;
ymn = y/maxVm;
zmn = z/maxVm;

% Moving Average filter///////////////////////////////
% Apply Hanning window
hanW = hanning(L+1);
xh = hanW.*(xmn-mean(xmn)); 
yh = hanW.*(ymn-mean(ymn)); 
zh = hanW.*(zmn-mean(zmn)); 
%Apply Moving Average Filter
num = 11;
for i = 1:num
    xa = movmean(xh, num);
    ya = movmean(yh, num); 
    za = movmean(zh, num);
end


% Moving Average Plot
figure;
plot(xa, '.');
hold on;
plot(ya, '.');
plot(za, '.');
title('Subject Moving Average)');
legend('x', 'y', 'z');

% Frequency Shit %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Raw/////////////////////////////
X = fft(z);
P2 = abs(X/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:(L/2))/L;
figure;

len = length(P1);
fAC = f(2:len);
P1AC = P1(2:len);

stem(fAC,P1AC) 
title('Single-Sided Amplitude Spectrum of X(t) (Raw)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

% Median/////////////////////////////
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
title('Single-Sided Amplitude Spectrum of X(t) (Median)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

% Moving Average /////////////////////////////
X = fft(za);
P2 = abs(X/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:(L/2))/L;
figure;

len = length(P1);
fAC = f(2:len);
P1AC = P1(2:len);

stem(fAC,P1AC) 
title('Single-Sided Amplitude Spectrum of X(t) (Median and MA)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

