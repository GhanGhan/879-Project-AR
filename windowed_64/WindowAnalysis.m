close all;
clc;
clear;
M = csvread('subject_data_1.csv', 1, 1);
row = 1000;
Section = M(row,:);
K = 64;
x = Section(1:K);
y = Section(K+1:2*K);
z = Section(2*K+1:3*K);

plot(x, '.');
hold on;
plot(y,'.');
plot(z, '.');