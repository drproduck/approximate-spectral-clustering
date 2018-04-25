function [U, S, V] = bask_doc_term(W, k, T, opts)
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


n = size(W, 1);
m = size(W, 2);

if (~exist('opts','var'))
   opts = [];
end  

if isfield(opts, 's')
    WN = W;
    s = opts.s;
    idx = zeros(n,s);
    for i = 1:s
        [~,idx(:,i)] = max(WN,[],2);
        temp = (idx(:,i)-1)*n+(1:n)';
        WN(temp) = 1e-100; 
    end
end
    

d1 = sum(W, 2);
d1 = max(d1, 1e-12);
d2 = sum(W, 1);
d2 = max(d2, 1e-12);

D1 = sparse(1:n, 1:n, d1.^(-0.5));
D2 = sparse(1:m, 1:m, d2.^(-0.5));

L = D1 * W * D2;

if isfield(opts, 'neigv')
    [U,S,V] = svds(L, opts.neigv);  
else
    [U,S,V] = svds(L, k);
end

% 2x
if T == 0
    U = D1 * U;
    V = D2 * V;
else
    U = D1 * U * S.^(T);
    V = D2 * V * S.^(T);
end
U(:,1) = [];
V(:,1) = [];
% U = max(U, 1e-12);
% V = max(V, 1e-12);
U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
V = bsxfun(@rdivide, V, sqrt(sum(V.^2, 2)));

end

