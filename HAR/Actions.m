function S =  Actions(M, S, ID)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

N = length(M);

for i = 1:N
    value = M(i);
    if value == 0
        S(ID, 8) = S(ID, 8) + 1;
    else
        S(ID, value) = S(ID, value) +1;
    end
end
end

