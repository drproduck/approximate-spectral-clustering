function [label, U, sigma, reps] = bask(fea, k, affinity, mode, t, opts)
%Bipartite Approximate Spectral Clustering
% fea: the dataset feature
% affinity: currently supports cosine and radial basis function (gaussian).
% mode: even_original: t is even, clustering original data's embedding
%       even_reps: t is even, clustering representative embedding, then
%                   associate to original data
%       odd_all: t is odd, clustering both original and representative
%                   data, only return labels of original data   
%       zero_original: basic specrtal clustering, clustering original data's
%                   embedding
%       zero_reps: basic spectral clustering, clustering representative
%                   embedding and associate to original data
% k: handpicked number of clusters
% r: number of representatives (d. 100)
% t: diffustion time (d. 0)
% sparse: enable sparsification, also set sparse parameter s = opts.sparse
% initRes/finalRes: number of restarts for initial/final Kmeans (d. 1/10)
% initIter/finalIter: number of maximum iterations for initial/final
% Kmeans(d. 5/100)
% the following 2 variables come from pre-processing k-means
% VAR: 1-by-r sum-of-squared-distances from member points to their centroids
% lbcount: size of each clusters

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
        else
            error('unsupported mode');
        end
    % default to k-means sampling
    else 
        [lb, reps] = litekmeans(fea, r, 'Distance', 'cosine', 'MaxIter', initIter, 'Replicates', initRes);
        lbcount = hist(lb, 1:r);
    end
    W = fea * reps';
    
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
    % if reps is provided (e.g in testing), should provide lbcount and VAR,
    % (not necessary) which come with k-means. See definitions at top
    if isfield(opts, 'reps')
        reps = opts.reps;
        % if lbcount is not input, default to 1
        if isfield(opts, 'lbcount')
            lbcount = opts.lbcount;
        else
            lbcount =  ones(1,r);
        end
        if isfield(opts, 'var')
            VAR = opts.var;
        end
        
    elseif isfield(opts, 'mode')
        if strcmp(opts.mode, 'kmeans')
        [lb, reps, ~, VAR] = litekmeans(fea, r, 'MaxIter', initIter, 'Replicates', initRes);
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
    elseif exist('VAR', 'var')
        sigma = mean(sqrt(VAR ./ lbcount));
    % default not recommended, should provide either sigma or VAR.
    else
        sigma = mean(mean(W));
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
        dump = exp(-dump/(2.0*sigma^2));
        Gidx = repmat((1:n)',1,s);
        Gjdx = idx;
        W = sparse(Gidx(:),Gjdx(:),dump(:),n,r);
        W = bsxfun(@times, W, lbcount);
    else
        if isfield(opts, 'kasp')
            [~,idx] = min(W,[],2);
        end
        W = exp(-W/(2.0*sigma^2));
        W = bsxfun(@times, W, lbcount);
    end
end

d1 = sum(W, 2);
d1 = max(d1, 1e-100);
d2 = sum(W, 1);
d2 = max(d2, 1e-100);

D1 = sparse(1:n, 1:n, d1.^(-0.5));
D2 = sparse(1:r, 1:r, d2.^(-0.5));

L = D1 * W * D2;

if isfield(opts, 'neigv')
    [U,S,V] = mySVD(L, opts.neigv);  
else
    [U,S,V] = mySVD(L, k);
end

finalIter = 100;
finalRes = 10;
if isfield(opts, 'finalIter')
    finalIter = opts.finalIter;
end
if isfield(opts, 'finalRes')
    finalRes = opts.finalRes;
end

% different ways to do clustering
if strcmp(mode, 'zero_original')
    U = D1 * U;
    U(:,1) = [];
    U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
    label = litekmeans(U, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
elseif strcmp(mode, 'zero_reps')
    V = D2 * V;
    V(:,1) = [];
    V = bsxfun(@rdivide, V, sqrt(sum(V.^2, 2)));
    reps_labels = litekmeans(V, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
    label = zeros(size(idx, 1), s);
    for i = 1:size(idx, 1)
        label(i) = reps_labels(idx(i));
    end
    
elseif strcmp(mode, 'even_original')
    U = D1 * U * S.^t;
    U(:,1) = [];
    U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
    label = litekmeans(U, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
    
elseif strcmp(mode, 'even_reps')
    V = D2 * V * S.^t;
    V(:,1) = [];
    V = bsxfun(@rdivide, V, sqrt(sum(V.^2, 2)));
    reps_labels = litekmeans(V, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
    label = zeros(size(idx, 1), s);
    for i = 1:size(idx, 1)
        label(i) = reps_labels(idx(i));
    end
elseif strcmp(mode, 'odd_all')
    U = D1 * U * S.^t;
    V = D2 * V * S.^t;
    W = [U;V];
    W = bsxfun(@rdivide, W, sqrt(sum(W.^2, 2))); 
    W(:,1) = [];
    all_label = litekmeans(W, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
    label = all_label(1:n);
end

end