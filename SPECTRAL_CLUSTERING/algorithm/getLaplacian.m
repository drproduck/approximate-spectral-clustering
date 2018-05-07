
function L = getLaplacian(A, tol, mode)

% compute normalized Laplacian of a matrix based on mode

[n,m] = size(A);
if strcmp(mode, 'symmetric')
    d1 = sum(A, 2);
    d1 = max(d1, tol);
    D = sparse(1:n,1:n,d1.^(-0.5));
    L = D*A*D;

elseif strcmp(mode, 'bipartite')
    d1 = sum(A, 2);
    d2 = sum(A, 1);
    d1 = max(d1, tol);
    d2 = max(d2, tol);
    D1 = sparse(1:n,1:n,d1.^(-0.5));
    D2 = sparse(1:m,1:m,d2.^(-0.5));
    L = D1*A*D2;
    
end
end
