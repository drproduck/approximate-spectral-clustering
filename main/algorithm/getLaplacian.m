function [L,D1,D2] = getLaplacian(A, tol, mode, varargin)
% getLaplacian compute normalized Laplacian of a matrix based on mode
%   A: input matrix
%   tol: tolerance constant, to "clip" degree away from 0
%   r: regularizer constant
%   mode: symmetric or bipartite
% return L: normalized Laplacian matrix
%        D1: row-sum ^ (-1/2) \in R^(n*n)
%        D2: col-sum ^ (-1/2) \in R^(m*m)
[n,m] = size(A);
s=sum(sum(A));

if strcmp(mode,'bipartite')
    r=[s/n,s/m];
elseif strcmp(mode,'symmetric')
    r=s/n;
end
p=inputParser;
addParameter(p,'r',r);
parse(p,varargin{:});
r=p.Results.r

if strcmp(mode, 'symmetric')
    d1 = sum(A, 2)+r;
    d1 = max(d1, tol);
    D = sparse(1:n,1:n,d1.^(-0.5));
    L = D*A*D;

elseif strcmp(mode, 'bipartite')
    d1 = sum(A, 2)+r(1);
    d2 = sum(A, 1)+r(2);
    d1 = max(d1, tol);
    d2 = max(d2, tol);
    D1 = sparse(1:n,1:n,d1.^(-0.5));
    D2 = sparse(1:m,1:m,d2.^(-0.5));
    L = D1*A*D2;
    
end
end
