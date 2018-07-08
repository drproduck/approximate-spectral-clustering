function [labels, U] = SC(fea, k, affinity, opts)

if (~exist('opts','var'))
   opts = [];
end

n = size(fea, 1);



random_index = randsample(n, r, false);
reps = fea(random_index, :)

if strcmp(affinity, 'cosine')
    W =  afcos(fea) - speye(size(fea,1), 1);
elseif strcmp(affinity, 'euclidian')
    W = EuDist2(fea);
elseif strcmp(affinity, 'gaussian')
    W = EuDist2(fea, fea, 0);
    if isfield(opts, 'sigma')
        sigma = opts.sigma;
    else sigma = mean(mean(W));
    end
    W = exp(-W/(2.0 * sigma^2)) - speye(size(fea, 1));
    
end

D = sum(W, 2);
D = max(D, 1e-12);
D = sparse(1:n, 1:n, D.^(-0.5));
L = D * W * D;


[U, ~, ~] = mySVD(L, k);
% U(:,1) = [];
U = normr(U);

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