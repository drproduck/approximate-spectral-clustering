function [labels, U] = SC(fea, k, affinity, opts)

if (~exist('opts','var'))
   opts = [];
end

n = size(fea, 1);

if strcmp(affinity, 'cosine')
    W =  afcos(fea);
elseif strcmp(affinity, 'euclidian')
    W = EuDist2(fea);
elseif strcmp(affinity, 'gaussian')
    W = EuDist2(fea, fea, 0);
    if isfield(opts, 'sigma')
        sigma = opts.sigma;
    else
        error('sigma must be specified')
    end
    W = exp(-W/(2.0 * sigma^2));
end
W = W - speye(n);

% sparse is the percent of non-zero elements to keep
if isfield(opts, 'sparse')
    s = opts.sparse;
    if isfield(opts, 'mode')
        mode = opts.mode;
    else
        mode = 'local';
    end
    if strcmp(mode, 'global')
        s = round(n^2 * s / 2, 0);
        L = tril(W, -1);
        [val, arg] = sort(L(:), 'descend');
        val = val(1:s);
        arg = arg(1:s);
        col_index = ceil(arg / n);
        row_index = arg - n * (col_index - 1);
        val = [val; val];
        col = [col_index; row_index];
        row = [row_index; col_index];
        W = sparse(row, col, val, n, n);
    elseif strcmp(mode, 'local')
        s = round(n * s, 0);
%         s = round(s/2, 0);
        maxStore = zeros(n, s);
        idxStore = zeros(n, s);
        for i = 1:s
            [maxStore(:,i), idxStore(:,i)] = max(W, [], 2);
            temp = (idxStore(:,i)-1)*n+(1:n)';
            W(temp) = 1e-100;
        end
        row_index = repmat((1:n)',s,1);
        col_index = idxStore(:);
        col = [col_index; row_index];
        row = [row_index; col_index];
        val = maxStore(:);
        val = [val;val];
        W = sparse(row, col, val, n, n);
    end
end

        
%njw
D = sum(W, 2);
D = max(D, 1e-12);
D = sparse(1:n, 1:n, D.^(-0.5));
L = D * W * D;


[U, ~, ~] = mySVD(L, k);
% U(:,1) = [];
U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));

MaxIter = 100;
if isfield(opts, 'MaxIter')
    MaxIter = opts.MaxIter;
end
Replicates = 10;
if isfield(opts, 'Replicates')
    Replicates = opts.Replicates;
end

labels = litekmeans(U, k, 'MaxIter', MaxIter, 'Replicates', Replicates);

end