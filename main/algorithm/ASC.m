function [labels, L] = ASC(fea, k, affinity, opts)
% assymetric spectral clustering: sparsifying the affinity matrix such that
% it is nomore symmetric, and normalize using bipartite model

if (~exist('opts','var'))
   opts = [];
end

n = size(fea, 1);

if strcmp(affinity, 'cosine')
    W =  afcos(fea) - speye(size(fea,1), 1);
elseif strcmp(affinity, 'euclidian')
    W = EuDist2(fea);
elseif strcmp(affinity, 'gaussian')
    W = EuDist2(fea, fea, 0);
    sigma = mean(mean(W));
    if isfield(opts, 'sigma')
        sigma = opts.sigma;
    end
    W = exp(-W/(2.0 * sigma^2)) - speye(size(fea, 1));
end

if isfield(opts, 'sparse')
    s = opts.sparse;
else
    s = round(sqrt(n), 0);
end

%sparsification
dump = zeros(n,s);
idx = dump;
for i = 1:s
    [dump(:,i),idx(:,i)] = max(W,[],2);
    temp = (idx(:,i)-1)*n+(1:n)';
    W(temp) = 1e-100; 
end
Gidx = repmat((1:n)',1,s);
Gjdx = idx;
W = sparse(Gidx(:),Gjdx(:),dump(:),n,n);    

%njw
D1 = sum(W, 2);
D1 = max(D1, 1e-12);
D2 = sum(W, 1);
D2 = max(D2, 1e-12);
D1 = sparse(1:n, 1:n, D1.^(-0.5));
D2 = sparse(1:n, 1:n, D2.^(-0.5));
L = D1 * W * D2;


[U, ~, ~] = mySVD(L, k);
% U(:,1) = [];
U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));

MaxIter = 10;
if isfield(opts, 'MaxIter')
    MaxIter = opts.MaxIter;
end
Replicates = 5;
if isfield(opts, 'Replicates')
    Replicates = opts.Replicates;
end

labels = litekmeans(U, k, 'MaxIter', MaxIter, 'Replicates', Replicates);

end