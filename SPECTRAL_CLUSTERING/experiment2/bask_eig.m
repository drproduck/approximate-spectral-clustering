function [U,S,V,idx,D1,D2] = bask_eig(fea, k, affinity, opts)
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
        W = bsxfun(@times, W, lbcount);
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
        W = bsxfun(@times, W, lbcount);
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

