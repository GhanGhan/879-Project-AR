clc;
clear;
close all;
%Constants
M = csvread('subject1.csv', 0, 1);
L = 128;     % Length of window
S = 50000;  % Starting index
N = S+L;   % Last index 
fc = 2;   % Cutoff frequency
fs = 52;  % Sampling frequency
fn = fs/2;  % Nyquist frequency
Order = 7;  %Order of Filter

hannW = hann(L);
%////////// SUBJECT 

% Raw Values
x = M(S:N,1);
y = M(S:N,2);
z = M(S:N,3);

% Raw Plot
figure;
plot(x, '.');
hold on;
plot(y, '.');
plot(z, '.');
title('Subject (RAW)');
legend('x', 'y', 'z');

% Median filter
x = movmean(x, 3); %medfilt1(x);
y = movmean(y, 3); %medfilt1(x);
z = movmean(z, 3); %medfilt1(x);

% Normalized Values
maxV = max([max(x) max(y) max(z)]);
xn = x/maxV;
yn = y/maxV;
zn = z/maxV;
% (Median) Raw Plot
figure;
plot(x, '.');
hold on;
plot(y, '.');
plot(z, '.');
title('Subject (Median RAW)');
legend('x', 'y', 'z');
% Apply Hanning window
hanW = hanning(L+1);
xh = hanW.*(x-mean(x)); 
yh = hanW.*(y-mean(y)); 
zh = hanW.*(z-mean(z)); 

% Butterworth Filter
[b, a] = butter(Order, fc/fn, 'low');

% Filtered Values
xf = filter(b,a, xh);
yf = filter(b,a, yh);
zf = filter(b,a, zh);

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
title('Subject (Butter Filtered and Normalized)');
legend('x', 'y', 'z');

% Frequency Shit
X_f = fft(zfn);
P2_f = abs(X_f/L);
P1_f = P2_f(1:L/2+1);
P1_f(2:end-1) = 2*P1_f(2:end-1);
f = fs*(0:(L/2))/L;
figure;

len = length(P1_f);
fAC = f(2:len);
P1AC_f = P1_f(2:len);

stem(fAC,P1AC_f) 
title('Single-Sided Amplitude Spectrum of X(t) (Filtered and Normalized)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

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

fprintf('Normalized mean value: %0.4f \n', mean(zn))
fprintf('Normalized standard dev: %0.4f \n', std(zn))
fprintf('Raw mean value: %0.4f \n', mean(z))
fprintf('Raw standard dev: %0.4f \n', std(z))
[v, i ] = max(P1AC_f);
i = fAC(i);
fprintf('Largest AC Frequency: %d \n', i); 
fprintf('Has magnitude: %0.5f \n', v);