function [dr] = outlier_remove(A, B, affinity)
n = size(A, 1);
m = size(B, 1);
if strcmp(affinity, 'cosine')
    imm = B' * ones(m,1);
    dr = A * imm;
    
elseif strcmp(affinity, 'euclidean')
    An = sum(A.^2, 2);
    
    %to avoid overflow
    Aminus = An - mean(An);
    
    AA = Aminus * m;
    imm = B' * ones(m,1);
    AB = A * imm;
    
    dr = AA  - 2 * AB;
end