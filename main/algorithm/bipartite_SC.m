function [label, kept_idx, U, reps] = biparite_SC(fea, u, k, r, s, t, affinity, varargin)
%LARGE SCALE SPECTRAL CLUSTERING USING DIFFUSION COORDINATE ON
%LANDMARK-BASED BIPARTITE GRAPH

%NOTE TO SELF: currently not multiplying matrix with lbcount

%REQUIRED:
%fea: the data in row-major order (i.e each datapoint is a row)
%k: number of clusters
%r: number of representatives (e.g 500)
%s: number of nearest landmarks to keep (e.g 5)
%t: diffusion time step (e.g 2)

%PARAMETER:
%distances to other points
%sigma: scaling factor for gaussian kernel.

%select_method: 
                %'random': pick landmarks uniformly random
                %'++': pick landmarks using kmeans++ weighting
                %'kmeans': pick landmarks as centers of a kmeans run
%embed_method:
                %'landmark': use right singular vector
                %'direct': use left singular vector
                %'coclustering': use both
%cluster_method: algorithms to partition embeddings
                %'kmeans':
                %'discretize:               
% initRes/finalRes: number of restarts for initial/final Kmeans (d. 1/10)
% initIter/finalIter: number of maximum iterations for initial/final


[n,m] = size(fea);

if (~exist('opts','var'))
   opts = [];
end

%% input parser
p = inputParser;
addParameter(p,'initIter',10);
addParameter(p,'initRes',1);
addParameter(p,'finalIter',100);
addParameter(p,'finalRes',10);
addParameter(p,'select_method','kmeans');
addParameter(p,'embed_method','landmark');
addParameter(p,'cluster_method','kmeans');
addParameter(p,'remove_outlier',[1,1]);
addParameter(p,'sigma',false);
addParameter(p,'fileid',1);
addParameter(p,'reps',false);
parse(p,varargin{:});

initIter = p.Results.initIter;
initRes = p.Results.initRes;
finalIter = p.Results.finalIter;
finalRes = p.Results.finalRes;
select_method = p.Results.select_method;
embed_method = p.Results.embed_method;
cluster_method = p.Results.cluster_method;
remove_outlier = p.Results.remove_outlier;
sigma = p.Results.sigma;
reps = p.Results.reps;
fileid = p.Results.fileid;

%% affinity
       
fprintf(fileid,'using gaussian affinity\n');

%select landmarks
fprintf(fileid,'selecting landmarks using ');
tic;
if strcmp(select_method, 'kmeans')
    fprintf(fileid,'kmeans...\n');
%         warning('off', 'stats:kmeans:FailedToConverge')
    [lb, reps, ~, VAR] = litekmeans(fea, r, 'MaxIter', initIter, 'Replicates', initRes,...
        'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100, 'clustersample', 0.1);
    lbcount = hist(lb, 1:r);

elseif strcmp(select_method, 'uniform')
    fprintf(fileid,'random sampling\n');
    reps = fea(randsample(n, r, false), :);

elseif strcmp(select_method, '++')
    fprintf(fileid,'D2 weight sampling\n');
    [~, reps] = kmeans(fea, r, 'MaxIter',0,'Replicates',1);
elseif strcmp(select_method, 'given') & reps ~= []
    fprintf(fileid,'given reps');
    reps = reps
else
    error('unsupported mode');
end
fprintf(fileid,'done in %.2f seconds\n',toc);

%determine sigma
if ~sigma
    if strcmp(select_method, 'kmeans')
        sigma = mean(sqrt(VAR ./ lbcount));

    elseif strcmp(select_method, 'random') || strcmp(select_method, '++')
        error('method random and ++ currently do not support finding sigma, please specify sigma in this function argument')
    end
end
    
W = EuDist2(fea, reps, 0);
[U,V] = decompose(W,k,length(reps),s,t,sigma);

%% cluster embeddings

if strcmp(embed_method, 'landmark')
    V(:,1) = [];
    V = V ./ sqrt(sum(V .^2, 2));
    reps_labels = litekmeans(V, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
    
    label = zeros(n, 1);
    for i = 1:n
        label(i) = reps_labels(idx(i));
    end
    
elseif strcmp(embed_method, 'direct')
    U(:,1) = [];
    U = U ./ sqrt(sum(U .^2, 2));
    label = litekmeans(U, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
    
end

end

function [U,V] = decompose(W,k,r,s,t,sigma)
[n,m] = size(W);
%sparse representation
if s > 0
    dump = zeros(n,s);
    idx = dump;
    for i = 1:s
        [dump(:,i),idx(:,i)] = min(W,[],2);
        temp = (idx(:,i)-1)*n+(1:n)';
        W(temp) = 1e100; 
    end

    dump = exp(-dump/(2.0*sigma^2));
    Gidx = repmat((1:n)',1,s);
    Gjdx = idx;
    W = sparse(Gidx(:),Gjdx(:),dump(:),n,r);

elseif s <= 0 % default to dense matrix
    fprintf(fileid,'default to dense matrix\n');
    if strcmp(embed_method, 'landmark')
        [~,idx] = min(W,[],2);
    end
    W = exp(-W/(2.0*sigma^2));
end

%% compute laplacian
d1 = sum(W, 2);
d2 = sum(W, 1);
d1 = max(d1, 1e-15);
d2 = max(d2, 1e-15);
D1 = sparse(1:n,1:n,d1.^(-0.5));
D2 = sparse(1:r,1:r,d2.^(-0.5));
L = D1*W*D2;
[U,S,V] = svds(L, k);

if t > 0
    U = D1 * U * S.^t;
    V = D2 * V * S.^t;
elseif t == 0
    U = D1 * U;
    V = D2 * V;
end


end