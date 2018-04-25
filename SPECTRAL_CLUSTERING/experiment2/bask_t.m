function [l1,l2,l3,l4,l5,l6,l7,l8, original_t, reps_t, all_t] = bask_t(fea, k, affinity, t, opts)
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
% reps: an k-by-r precalculated set of cluster centroids1

t0 = cputime;
n = size(fea, 1);

if (~exist('opts','var'))
   opts = [];
end

% number of representative points
r = 100;
if isfield(opts, 'r')
    r = opts.r;
end    
    
if strcmp(affinity, 'gaussian')
    % if reps is provided (e.g in testing), should provide lbcount and VAR,
    % (not necessary) which come with k-means. See definitions at top

    reps = opts.reps;
    lbcount = opts.lbcount;
    W = EuDist2(fea, reps, 0);
    
    if isfield(opts, 'sigma')
        sigma = opts.sigma;
    else
        sigma = mean(mean(W));
    end
 
    % sparse representation
    if isfield(opts, 's')
        s = opts.s;
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
        % W = bsxfun(@times, W, lbcount);
    end
elseif strcmp(affinity, 'cosine')
    reps = opts.reps;
    lbcount = opts.lbcount;
    W = fea * reps';
    if isfield(opts, 's')
        s = opts.s;
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
        % W = bsxfun(@times, W, lbcount);
    end
else
    error('unsupported affinity function')
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
base_t = cputime - t0;

% only for testing
UN = U;
VN = V;
% different ways to do clustering


% zero_original
t0 = cputime;
U = D1 * UN;
U(:,1) = [];
U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
l1 = litekmeans(U, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
original_t = cputime - t0 + base_t;
    
% zero_reps
t0 = cputime;
V = D2 * VN;
V(:,1) = [];
V = bsxfun(@rdivide, V, sqrt(sum(V.^2, 2)));
reps_label = litekmeans(V, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
l2 = zeros(n, 1);
for i = 1:n
    % give labels of r-nearest reps, pick label with the most occurence. 
    can_reps_label = reps_label(idx(i,:));
    [~, m] = max(hist(can_reps_label, 1:k));
    l2(i) = m;
end
reps_t = cputime - t0 + base_t;

l3 = zeros(n, t);
l4 = zeros(n, t);
l5 = zeros(n, t);
for T = 1:t
    % even_original
    U = D1 * UN * S.^(2*T);
    U(:,1) = [];
    U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
    l3(:,T) = litekmeans(U, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);

    % even_reps
    V = D2 * VN * S.^(2*T);
    V(:,1) = [];
    V = bsxfun(@rdivide, V, sqrt(sum(V.^2, 2)));
    reps_label = litekmeans(V, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
    for i = 1:n
        % give labels of r-nearest reps, pick label with the most occurence. 
        can_reps_label = reps_label(idx(i,:));
        [~, m] = max(hist(can_reps_label, 1:k));
        l4(i,T) = m;
    end

    % odd_all
    U = D1 * UN * S.^(2*T-1);
    V = D2 * VN * S.^(2*T-1);
    W = [U;V];
    W(:,1) = [];
    W = bsxfun(@rdivide, W, sqrt(sum(W.^2, 2))); 
    all_label = litekmeans(W, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
    l5(:,T) = all_label(1:n);
end
%zero_all
t0 = cputime;
U = D1 * UN;
V = D2 * VN;
W = [U;V];
W(:,1) = [];
W = bsxfun(@rdivide, W, sqrt(sum(W.^2, 2))); 
all_label = litekmeans(W, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
l6 = all_label(1:n);
all_t = cputime - t0 + base_t;

%zero_no_d_original
U = UN;
U(:,1) = [];
U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
l7 = litekmeans(U, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
%zero_no_d_reps
V = VN;
V(:,1) = [];
V = bsxfun(@rdivide, V, sqrt(sum(V.^2, 2)));
reps_label = litekmeans(V, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
l8 = zeros(n, 1);
for i = 1:n
    % give labels of r-nearest reps, pick label with the most occurence. 
    can_reps_label = reps_label(idx(i,:));
    [~, m] = max(hist(can_reps_label, 1:k));
    l8(i) = m;
end




end
