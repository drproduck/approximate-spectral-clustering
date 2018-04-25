function [l3,l4,l5,dhillon_acc, t3,t4,t5,dhillon_t] = bask_news(W, k, t, opts)
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
% sparse: enable sparsification, also set sparse parameter s = opts.sparse
% initRes/finalRes: number of restarts for initial/final Kmeans (d. 1/10)
% initIter/finalIter: number of maximum iterations for initial/final
% Kmeans(d. 5/100)
% the following 2 variables come from pre-processing k-means
% VAR: 1-by-r sum-of-squared-distances from member points to their centroids
% lbcount: size of each clusters

%TESTING OPTIONS
% reps: an k-by-r precalculated set of cluster centroids


t0 =  cputime;
n = size(W, 1);
m = size(W, 2);

% if isfield(opts, 's')
%     WN = W;
%     s = opts.s;
%     idx = zeros(n,s);
%     for i = 1:s
%         [~,idx(:,i)] = max(WN,[],2);
%         temp = (idx(:,i)-1)*n+(1:n)';
%         WN(temp) = 1e-100; 
%     end
% end


if (~exist('opts','var'))
   opts = [];
end  
    

d1 = sum(W, 2);
d1 = max(d1, 1e-100);
d2 = sum(W, 1);
d2 = max(d2, 1e-100);

D1 = sparse(1:n, 1:n, d1.^(-0.5));
D2 = sparse(1:m, 1:m, d2.^(-0.5));

L = D1 * W * D2;

if isfield(opts, 'neigv')
    [U,S,V] = svds(L, opts.neigv);  
else
    [U,S,V] = svds(L, k);
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

l3 = zeros(n, t);
t3 = zeros(t,1);
l4 = zeros(n, t);
t4 = zeros(t,1);
l5 = zeros(n, t);
t5 = zeros(t,1);
for T = 1:t
    % 2x
    t0 = cputime;
    U = D1 * UN * S.^(2*T);
    U(:,1) = [];
    U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
    l3(:,T) = litekmeans(U, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
    t3(T) = cputime - t0 + base_t;
    
    % 2y
    t0 = cputime;
    V = D2 * VN * S.^(2*T);
    V(:,1) = [];
    V = bsxfun(@rdivide, V, sqrt(sum(V.^2, 2)));
    reps_label = litekmeans(V, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
    for i = 1:n
        % give labels of r-nearest reps, pick label with the most occurence. 
        can_reps_label = find(W(i,:) ~= 0);
        for j = 1:size(can_reps_label,2)
            can_reps_label(j) = reps_label(can_reps_label(j));
        end
        [~, m] = max(hist(can_reps_label, 1:k));
        l4(i,T) = m;
    end
    t4(T) = cputime - t0 + base_t;

    % 1
    t0 = cputime;
    U = D1 * UN * S.^(2*T-1);
    V = D2 * VN * S.^(2*T-1);
    W = [U;V];
    W(:,1) = [];
    W = bsxfun(@rdivide, W, sqrt(sum(W.^2, 2))); 
    all_label = litekmeans(W, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
    l5(:,T) = all_label(1:n);
    t5(T) = cputime - t0 + base_t;
end

%Dhillon
t0 = cputime;
U = D1 * UN;
V = D2 * VN;
W = [U;V];
W(:,1) = [];
W = bsxfun(@rdivide, W, sqrt(sum(W.^2, 2))); 
all_label = litekmeans(W, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
dhillon_acc = all_label(1:n);
dhillon_t = cputime - t0 + base_t;

% cSPEC
% t0 = cputime;
% U = UN;
% U(:,1) = [];
% U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
% cspec_acc = litekmeans(U, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
% cspec_t = cputime - t0 + base_t;

end

