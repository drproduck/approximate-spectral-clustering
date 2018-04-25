function [label, U, sigma] = BiAS(fea, k, affinity, opts)
%Bipartite Approximate Spectral Clustering

% affinity: currently supports cosine and radial basis function (gaussian).
% sigma: scaling factor for gaussian kernel. Default is computed as
% mean(mean(distance_matrix))
% k: handpicked number of clusters
% r: number of representatives (d. 100)
% t: diffustion time (d. 0)
% sparse: enable sparsification, also set sparse parameter l = opts.sparse
% mode: support choosing representative points by 'kmeans' or 'uniform'
% initRes/finalRes: number of restarts for initial/final Kmeans (d. 1/10)
% initIter/finalIter: number of maximum iterations for initial/final
% Kmeans(d. 5/100)

%TESTING OPTIONS
% reps: an k-by-r precalculated set of cluster centroids

n = size(fea, 1);

if (~exist('opts','var'))
   opts = [];
end

% number of representative points
r = 100;
if isfield(opts, 'r')
    r = opts.r;
end

initIter = 10;
initRes = 1;
if isfield(opts, 'initIter')
    initIter = opts.initIter;
end
if isfield(opts, 'initRes')
    initRes = opts.initRes;
end
    
if strcmp(affinity, 'cosine')
    fea = bsxfun(@rdivide, fea, sqrt(sum(fea.^2, 2)));
   
    if isfield(opts, 'reps')
        reps = opts.reps;
        % if lbcount is not input, default to not multiply
        if isfield(opts, 'lbcount')
            lbcount = opts.lbcount;
        else 
            lbcount = ones(1,r);
        end
        
    elseif isfield(opts, 'mode')
        if strcmp(opts.mode, 'kmeans') 
            [lb, reps] = litekmeans(fea, r, 'Distance', 'cosine', 'MaxIter', initIter, 'Replicates', initRes);
            lbcount = hist(lb, 1:r);
        elseif strcmp(opts.mode, 'uniform')
            reps = fea(randsample(n, r, false), :);
            lbcount = ones(1,r);
        else
            error('unsupported mode');
        end
    % default to k-means sampling
    else 
        [lb, reps] = litekmeans(fea, r, 'Distance', 'cosine', 'MaxIter', initIter, 'Replicates', initRes);
        lbcount = hist(lb, 1:r);
    end
    W = fea * reps';
    W = bsxfun(@times, W, lbcount);
    
    if isfield(opts, 'sparse')
        s = opts.sparse;
        dump = zeros(n,s);
        idx = dump;
        for i = 1:s
            [dump(:,i),idx(:,i)] = max(W,[],2);
            temp = (idx(:,i)-1)*n+(1:n)';
            W(temp) = 1e-100; 
        end
        Gidx = repmat((1:n)',1,s);
        Gjdx = idx;
        W = sparse(Gidx(:),Gjdx(:),dump(:),n,r);
        W = bsxfun(@times, W, lbcount);
    elseif isfield(opts, 'kasp')
        [~,idx] = max(W,[],2);
    end 
    
elseif strcmp(affinity, 'gaussian')
    if isfield(opts, 'reps')
        reps = opts.reps;
        % if lbcount is not input, default to not multiply
        if isfield(opts, 'lbcount')
            lbcount = opts.lbcount;
        else
            lbcount =  ones(1,r);
        end
    elseif isfield(opts, 'mode')
        if strcmp(opts.mode, 'kmeans')
        [lb, reps] = litekmeans(fea, r, 'MaxIter', initIter, 'Replicates', initRes);
        lbcount = hist(lb, 1:r);
        elseif strcmp(opts.mode, 'uniform')
            reps = fea(randsample(n, r, false), :);
            lbcount = ones(1,r);
        else
            error('unsupported mode');
        end
    % default to k-means sampling
    else 
        [lb, reps, ~, VAR] = litekmeans(fea, r, 'MaxIter', initIter, 'Replicates', initRes);
        lbcount = hist(lb, 1:r);
    end
    W = EuDist2(fea, reps, 0);
    
    if isfield(opts, 'sigma')
        sigma = opts.sigma;
    else
        sigma = sqrt(VAR ./ lbcount);
    end
 
    % sparse representation
    if isfield(opts, 'sparse')
        s = opts.sparse;
        dump = zeros(n,s);
        idx = dump;
        for i = 1:s
            [dump(:,i),idx(:,i)] = min(W,[],2);
            temp = (idx(:,i)-1)*n+(1:n)';
            W(temp) = 1e100; 
        end

        % manipulate index to efficiently create sparse matrix Z
        % Z is now (normalized to sum 1) smallest r landmarks in each row
        Gidx = repmat((1:n)',1,s);
        Gjdx = idx;
        W = sparse(Gidx(:),Gjdx(:),dump(:),n,r);
        denom = 2.0*sigma.^2;
        W = exp(-W./denom);
    else
        if isfield(opts, 'kasp')
            [~,idx] = min(W,[],2);
        end
        denom = 2.0*sigma.^2;
        W = exp(-W./denom);
        W = bsxfun(@times, W, lbcount);
    end
end

d1 = sum(W, 2);
d1 = max(d1, 1e-12);
d2 = sum(W, 1);
d2 = max(d2, 1e-12);

D1 = sparse(1:n, 1:n, d1.^(-0.5));
D2 = sparse(1:r, 1:r, d2.^(-0.5));

L = D1 * W * D2;

[U,S,V] = mySVD(L, k+1);

if isfield(opts, 't')
    t = opts.t;
    U = D1 * U * S.^(2*t);
    V = D2 * V * S.^(2*t);
end

U(:,1) = [];
V(:,1) = [];

% U = U ./ sqrt(sum(U .^2, 2));
% V = V ./ sqrt(sum(V .^2, 2));
U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
V = bsxfun(@rdivide, V, sqrt(sum(V.^2, 2)));

finalIter = 100;
finalRes = 10;
if isfield(opts, 'finalIter')
    finalIter = opts.finalIter;
end
if isfield(opts, 'finalRes')
    finalRes = opts.finalRes;
end

if isfield(opts, 'kasp')
    reps_labels = litekmeans(V, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);

    label = zeros(size(idx, 1), 1);
    for i = 1:size(idx, 1)
        label(i) = reps_labels(idx(i));
    end
else
    label = litekmeans(U, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
end

end