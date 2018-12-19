M = csvread('subject5.csv', 0, 1);
close all;

x = M(:,1);
y = M(:,2);
z = M(:,3);
l = M(:,4);

%XYZ AXIS
subplot(4, 1, 1)
plot(x)
legend('x')
xlabel('time index')
ylabel('acceleration')

subplot(4, 1, 2)
plot(y)
legend('y')
xlabel('time index')
ylabel('acceleration')

subplot(4, 1, 3)
plot(z)
legend('z')
xlabel('time index')
ylabel('acceleration')

subplot(4, 1, 4)
plot(l)
legend('class')
xlabel('time index')
ylabel('Label')

%%///////////////Walking vs walking and talking
%Walking and Talking
figure;
s = 81000;
e = 81500;
subplot(3, 1, 1)
plot(x(s:e))
legend('x')
xlabel('time index')
ylabel('acceleration')

subplot(3, 1, 2)
plot(y(s:e))
legend('y')
xlabel('time index')
ylabel('acceleration')

subplot(3, 1, 3)
plot(z(s:e))
legend('z')
xlabel('time index')
ylabel('acceleration')

%Walking
figure;
s = 63000;
e = 63500;
subplot(3, 1, 1)
plot(x(s:e))
legend('x')
xlabel('time index')
ylabel('acceleration')

subplot(3, 1, 2)
plot(y(s:e))
legend('y')
xlabel('time index')
ylabel('acceleration')

subplot(3, 1, 3)
plot(z(s:e))
legend('z')
xlabel('time index')
ylabel('acceleration')

