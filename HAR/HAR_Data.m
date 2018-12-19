%Number of each action from every subject
% 1 2 3 4 5 6 7 0
Subjects = zeros(15,8);


M1 = csvread('subject1.csv', 0, 4);
Subjects = Actions(M1, Subjects, 1);
R = 1;
total1 = sum(Subjects(R,:));
L1 = 100*Subjects(R,1)/total1; L2 = 100*Subjects(R,2)/total1; L3 = 100*Subjects(R,3)/total1;
L4 = 100*Subjects(R,4)/total1; L5 = 100*Subjects(R,5)/total1; L6 = 100*Subjects(R,6)/total1;
L7 = 100*Subjects(R,7)/total1;
fprintf('Subject 1 \n');
fprintf('1: %0.2f, 2: %0.2f, 3: %0.2f, 4: %0.2f, 5: %0.2f, 6: %0.2f, 7: %0.2f \n', L1, L2, L3, L4, L5, L6, L7);



M2 = csvread('subject2.csv', 0, 4);
Subjects = Actions(M2, Subjects, 2);
R = 2;
total1 = sum(Subjects(R,:));
L1 = 100*Subjects(R,1)/total1; L2 = 100*Subjects(R,2)/total1; L3 = 100*Subjects(R,3)/total1;
L4 = 100*Subjects(R,4)/total1; L5 = 100*Subjects(R,5)/total1; L6 = 100*Subjects(R,6)/total1;
L7 = 100*Subjects(R,7)/total1;
fprintf('Subject 2 \n');
fprintf('1: %0.2f, 2: %0.2f, 3: %0.2f, 4: %0.2f, 5: %0.2f, 6: %0.2f, 7: %0.2f \n', L1, L2, L3, L4, L5, L6, L7);

M3 = csvread('subject3.csv', 0, 4);
Subjects = Actions(M3, Subjects, 3);
R = 3;
total1 = sum(Subjects(R,:));
L1 = 100*Subjects(R,1)/total1; L2 = 100*Subjects(R,2)/total1; L3 = 100*Subjects(R,3)/total1;
L4 = 100*Subjects(R,4)/total1; L5 = 100*Subjects(R,5)/total1; L6 = 100*Subjects(R,6)/total1;
L7 = 100*Subjects(R,7)/total1;
fprintf('Subject 3 \n');
fprintf('1: %0.2f, 2: %0.2f, 3: %0.2f, 4: %0.2f, 5: %0.2f, 6: %0.2f, 7: %0.2f \n', L1, L2, L3, L4, L5, L6, L7);

M4 = csvread('subject4.csv', 0, 4);
Subjects = Actions(M4, Subjects, 4);
R = 4;
total1 = sum(Subjects(R,:));
L1 = 100*Subjects(R,1)/total1; L2 = 100*Subjects(R,2)/total1; L3 = 100*Subjects(R,3)/total1;
L4 = 100*Subjects(R,4)/total1; L5 = 100*Subjects(R,5)/total1; L6 = 100*Subjects(R,6)/total1;
L7 = 100*Subjects(R,7)/total1;
fprintf('Subject 4 \n');
fprintf('1: %0.2f, 2: %0.2f, 3: %0.2f, 4: %0.2f, 5: %0.2f, 6: %0.2f, 7: %0.2f \n', L1, L2, L3, L4, L5, L6, L7);

M5 = csvread('subject5.csv', 0, 4);
Subjects = Actions(M5, Subjects, 5);
R = 5;
total1 = sum(Subjects(R,:));
L1 = 100*Subjects(R,1)/total1; L2 = 100*Subjects(R,2)/total1; L3 = 100*Subjects(R,3)/total1;
L4 = 100*Subjects(R,4)/total1; L5 = 100*Subjects(R,5)/total1; L6 = 100*Subjects(R,6)/total1;
L7 = 100*Subjects(R,7)/total1;
fprintf('Subject 5 \n');
fprintf('1: %0.2f, 2: %0.2f, 3: %0.2f, 4: %0.2f, 5: %0.2f, 6: %0.2f, 7: %0.2f \n', L1, L2, L3, L4, L5, L6, L7);

M6 = csvread('subject6.csv', 0, 4);
Subjects = Actions(M6, Subjects, 6);
R = 6;
total1 = sum(Subjects(R,:));
L1 = 100*Subjects(R,1)/total1; L2 = 100*Subjects(R,2)/total1; L3 = 100*Subjects(R,3)/total1;
L4 = 100*Subjects(R,4)/total1; L5 = 100*Subjects(R,5)/total1; L6 = 100*Subjects(R,6)/total1;
L7 = 100*Subjects(R,7)/total1;
fprintf('Subject 6 \n');
fprintf('1: %0.2f, 2: %0.2f, 3: %0.2f, 4: %0.2f, 5: %0.2f, 6: %0.2f, 7: %0.2f \n', L1, L2, L3, L4, L5, L6, L7);

M7 = csvread('subject7.csv', 0, 4);
Subjects = Actions(M7, Subjects, 7);
R = 7;
total1 = sum(Subjects(R,:));
L1 = 100*Subjects(R,1)/total1; L2 = 100*Subjects(R,2)/total1; L3 = 100*Subjects(R,3)/total1;
L4 = 100*Subjects(R,4)/total1; L5 = 100*Subjects(R,5)/total1; L6 = 100*Subjects(R,6)/total1;
L7 = 100*Subjects(R,7)/total1;
fprintf('Subject 7 \n');
fprintf('1: %0.2f, 2: %0.2f, 3: %0.2f, 4: %0.2f, 5: %0.2f, 6: %0.2f, 7: %0.2f \n', L1, L2, L3, L4, L5, L6, L7);

M8 = csvread('subject8.csv', 0, 4);
Subjects = Actions(M8, Subjects, 8);
R = 8;
total1 = sum(Subjects(R,:));
L1 = 100*Subjects(R,1)/total1; L2 = 100*Subjects(R,2)/total1; L3 = 100*Subjects(R,3)/total1;
L4 = 100*Subjects(R,4)/total1; L5 = 100*Subjects(R,5)/total1; L6 = 100*Subjects(R,6)/total1;
L7 = 100*Subjects(R,7)/total1;
fprintf('Subject 8 \n');
fprintf('1: %0.2f, 2: %0.2f, 3: %0.2f, 4: %0.2f, 5: %0.2f, 6: %0.2f, 7: %0.2f \n', L1, L2, L3, L4, L5, L6, L7);

M9 = csvread('subject9.csv', 0, 4);
Subjects = Actions(M9, Subjects, 9);
R = 9;
total1 = sum(Subjects(R,:));
L1 = 100*Subjects(R,1)/total1; L2 = 100*Subjects(R,2)/total1; L3 = 100*Subjects(R,3)/total1;
L4 = 100*Subjects(R,4)/total1; L5 = 100*Subjects(R,5)/total1; L6 = 100*Subjects(R,6)/total1;
L7 = 100*Subjects(R,7)/total1;
fprintf('Subject 9 \n');
fprintf('1: %0.2f, 2: %0.2f, 3: %0.2f, 4: %0.2f, 5: %0.2f, 6: %0.2f, 7: %0.2f \n', L1, L2, L3, L4, L5, L6, L7);

M10 = csvread('subject10.csv', 0, 4);
Subjects = Actions(M10, Subjects, 10);
R = 10;
total1 = sum(Subjects(R,:));
L1 = 100*Subjects(R,1)/total1; L2 = 100*Subjects(R,2)/total1; L3 = 100*Subjects(R,3)/total1;
L4 = 100*Subjects(R,4)/total1; L5 = 100*Subjects(R,5)/total1; L6 = 100*Subjects(R,6)/total1;
L7 = 100*Subjects(R,7)/total1;
fprintf('Subject 10 \n');
fprintf('1: %0.2f, 2: %0.2f, 3: %0.2f, 4: %0.2f, 5: %0.2f, 6: %0.2f, 7: %0.2f \n', L1, L2, L3, L4, L5, L6, L7);

M11 = csvread('subject11.csv', 0, 4);
Subjects = Actions(M11, Subjects, 11);
R = 11;
total1 = sum(Subjects(R,:));
L1 = 100*Subjects(R,1)/total1; L2 = 100*Subjects(R,2)/total1; L3 = 100*Subjects(R,3)/total1;
L4 = 100*Subjects(R,4)/total1; L5 = 100*Subjects(R,5)/total1; L6 = 100*Subjects(R,6)/total1;
L7 = 100*Subjects(R,7)/total1;
fprintf('Subject 11 \n');
fprintf('1: %0.2f, 2: %0.2f, 3: %0.2f, 4: %0.2f, 5: %0.2f, 6: %0.2f, 7: %0.2f \n', L1, L2, L3, L4, L5, L6, L7);

M12 = csvread('subject12.csv', 0, 4);
Subjects = Actions(M12, Subjects, 12);
R = 12;
total1 = sum(Subjects(R,:));
L1 = 100*Subjects(R,1)/total1; L2 = 100*Subjects(R,2)/total1; L3 = 100*Subjects(R,3)/total1;
L4 = 100*Subjects(R,4)/total1; L5 = 100*Subjects(R,5)/total1; L6 = 100*Subjects(R,6)/total1;
L7 = 100*Subjects(R,7)/total1;
fprintf('Subject 12 \n');
fprintf('1: %0.2f, 2: %0.2f, 3: %0.2f, 4: %0.2f, 5: %0.2f, 6: %0.2f, 7: %0.2f \n', L1, L2, L3, L4, L5, L6, L7);

M13 = csvread('subject13.csv', 0, 4);
Subjects = Actions(M13, Subjects, 13);
R = 13;
total1 = sum(Subjects(R,:));
L1 = 100*Subjects(R,1)/total1; L2 = 100*Subjects(R,2)/total1; L3 = 100*Subjects(R,3)/total1;
L4 = 100*Subjects(R,4)/total1; L5 = 100*Subjects(R,5)/total1; L6 = 100*Subjects(R,6)/total1;
L7 = 100*Subjects(R,7)/total1;
fprintf('Subject 13 \n');
fprintf('1: %0.2f, 2: %0.2f, 3: %0.2f, 4: %0.2f, 5: %0.2f, 6: %0.2f, 7: %0.2f \n', L1, L2, L3, L4, L5, L6, L7);

M14 = csvread('subject14.csv', 0, 4);
Subjects = Actions(M14, Subjects, 14);
R = 14;
total1 = sum(Subjects(R,:));
L1 = 100*Subjects(R,1)/total1; L2 = 100*Subjects(R,2)/total1; L3 = 100*Subjects(R,3)/total1;
L4 = 100*Subjects(R,4)/total1; L5 = 100*Subjects(R,5)/total1; L6 = 100*Subjects(R,6)/total1;
L7 = 100*Subjects(R,7)/total1;
fprintf('Subject 14 \n');
fprintf('1: %0.2f, 2: %0.2f, 3: %0.2f, 4: %0.2f, 5: %0.2f, 6: %0.2f, 7: %0.2f \n', L1, L2, L3, L4, L5, L6, L7);

M15 = csvread('subject15.csv', 0, 4);
Subjects = Actions(M15, Subjects, 15);
R = 15;
total1 = sum(Subjects(R,:));
L1 = 100*Subjects(R,1)/total1; L2 = 100*Subjects(R,2)/total1; L3 = 100*Subjects(R,3)/total1;
L4 = 100*Subjects(R,4)/total1; L5 = 100*Subjects(R,5)/total1; L6 = 100*Subjects(R,6)/total1;
L7 = 100*Subjects(R,7)/total1;
fprintf('Subject 15 \n');
fprintf('1: %0.2f, 2: %0.2f, 3: %0.2f, 4: %0.2f, 5: %0.2f, 6: %0.2f, 7: %0.2f \n', L1, L2, L3, L4, L5, L6, L7);


I1 = find(M1 == 0);
I2 = find(M2 == 0);
I3 = find(M3 == 0);
I4 = find(M4 == 0);
I5 = find(M5 == 0);
I6 = find(M6 == 0);
I7 = find(M7 == 0);
I8 = find(M8 == 0);
I9 = find(M9 == 0);
I10 = find(M10 == 0);
I11 = find(M11 == 0);
I12 = find(M12 == 0);
I13 = find(M13 == 0);
I14 = find(M14 == 0);
I15 = find(M15 == 0);
