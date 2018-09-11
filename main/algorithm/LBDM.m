function [label, kept_idx, U, reps] = LBDM(fea, k, r, s, t, affinity, varargin)
%LARGE SCALE SPECTRAL CLUSTERING USING DIFFUSION COORDINATE ON
%LANDMARK-BASED BIPARTITE GRAPH

%NOTE TO SELF: currently not multiplying matrix with lbcount

%REQUIRED:
%fea: the data in row-major order (i.e each datapoint is a row)
%k: number of clusters
%r: number of representatives (e.g 500)
%s: number of nearest landmarks to keep (e.g 5)
%t: diffusion time step (e.g 2)
%affinity:
                %'gaussian'
                %'cosine' (will normalize features first)

%PARAMETER:
%remove_outlier: remove a subset of outliers based on point's total
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
fileid = p.Results.fileid;

    
%% affinity

%1 cosine

if strcmp(affinity, 'cosine')
    fprintf(fileid,'using cosine affinity\n');
    fprintf(fileid,'normalizing features...\n');
    fea = fea ./ sqrt(sum(fea.^2, 2));
    
    %remove outlier
    if ((0 < remove_outlier(1)) && (remove_outlier(1) < 1)) || ((0 < remove_outlier(2)) && (remove_outlier(2) < 1))
        tic;
        fprintf(fileid,'removing outliers by finding row-sums...\n');
        d = fea * (fea' * ones(n,1));
        [val, amax] = sort(d, 'descend');
        remove_high = remove_outlier(1);
        remove_low = remove_outlier(2);
        if (0 < remove_high) && (remove_high < 1)
            high = floor(n * remove_high);
        else
            high = 1;
        end
        if (0 < remove_low) && (remove_low < 1)
            low = n - ceil(n * remove_low);
        else 
            low = n;
        end
        kept_idx = amax(high:low);
        no_kept = size(kept_idx,1);
        plot(1:n,val);
        fprintf(fileid,'remove %.0f%% of dataset...\n',(1 - no_kept / n) *100);
        fprintf(fileid,'removing outliers done in %.2f seconds\n', toc);
    else
        no_kept = n;
        kept_idx = 1:n;
     
    end
        
    %select landmarks
    fprintf(fileid,'selecting landmarks using ');
    tic;
    if strcmp(select_method, 'kmeans') 
        fprintf(fileid,'kmeans...\n');
        [lb, reps] = litekmeans(fea(kept_idx,:), r, 'Distance', 'cosine', 'MaxIter', initIter, 'Replicates', initRes,...
            'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100, 'clustersample', 0.1);
        lbcount = hist(lb, 1:r); %#ok<NASGU>
    elseif strcmp(select_method, 'uniform')
        fprintf(fileid,'random sampling...\n');
        reps = fea(kept_idx(randsample(no_kept, r, false)), :);
    else
        error('unsupported mode');
    end
    
    W = fea * reps';
    fprintf(fileid,'done in %.2f seconds\n', toc);
    
    %construct A
    fprintf(fileid,'constructing sparse A...\n');
    tic;
    if s > 0
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
        %remove outliers
        W = W(kept_idx,:);
    fprintf(fileid,'done in %.2f seconds\n',toc);   
    elseif s <= 0 % default to dense matrix
        fprintf(fileid,'Default to dense martix\n');
        if strcmp(embed_method, 'landmark')
            [~,idx] = min(W,[],2);
            %remove outliers
            W = W(kept_idx,:);
        end
    end
       
%2 gaussian
elseif strcmp(affinity, 'gaussian')
    fprintf(fileid,'using gaussian affinity\n');
    %remove outlier
    if ((0 < remove_outlier(1)) && (remove_outlier(1) < 1)) || ((0 < remove_outlier(2)) && (remove_outlier(2) < 1))
        tic;
        fprintf(fileid,'removing outliers by finding row-sums...\n');
        An = sum(fea.^2, 2);
        Aminus = An - mean(An); %to avoid overflow  
        AA = Aminus * n;
        AB = fea * (fea' * ones(n,1));
        d = AA  - 2 * AB;
        [val, amax] = sort(d, 'descend');
        remove_high = remove_outlier(1);
        remove_low = remove_outlier(2);
        if (0 < remove_high) && (remove_high < 1)
            high = floor(n * remove_high);
        else 
            high = 1;
        end
        if (0 < remove_low) && (remove_low < 1)
            low = n - ceil(n * remove_low);
        else
            low = n;
        end
        kept_idx = amax(high:low);
        no_kept = size(kept_idx,1);
        plot(1:n,val);
        fprintf(fileid,'remove %.0f%% of dataset...\n',(no_kept / n) *100);
        fprintf(fileid,'removing outliers done in %.2f seconds\n', toc);
    else
        no_kept = n;
        kept_idx = 1:n;
    end
    
    %select landmarks
    fprintf(fileid,'selecting landmarks using ');
    tic;
    if strcmp(select_method, 'kmeans')
        fprintf(fileid,'kmeans...\n');
%         warning('off', 'stats:kmeans:FailedToConverge')
        [lb, reps, ~, VAR] = litekmeans(fea(kept_idx,:), r, 'MaxIter', initIter, 'Replicates', initRes,...
            'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100, 'clustersample', 0.1);
        lbcount = hist(lb, 1:r);
    
    elseif strcmp(select_method, 'uniform')
        fprintf(fileid,'random sampling\n');
        reps = fea(kept_idx(randsample(no_kept, r, false)), :);
       
    elseif strcmp(select_method, '++')
        fprintf(fileid,'D2 weight sampling\n');
        [~, reps] = kmeans(fea_kept, r, 'MaxIter',0,'Replicates',1);
    else
        error('unsupported mode');
    end
    fprintf(fileid,'done in %.2f seconds\n',toc);
    W = EuDist2(fea, reps, 0);
    
    %determine sigma
    if ~sigma
        if strcmp(select_method, 'kmeans')
            sigma = mean(sqrt(VAR ./ lbcount));

        elseif strcmp(select_method, 'random') || strcmp(select_method, '++')
            error('method random and ++ currently do not support finding sigma, please specify sigma in this function argument')
        end
    end
    
    fprintf(fileid,'using sigma = %.2f\n',sigma);
 
    %sparse representation
    fprintf(fileid,'constructing sparse A...\n');
    tic;
    if s > 0
        dump = zeros(n,s);
        idx = dump;
        for i = 1:s
            [dump(:,i),idx(:,i)] = min(W,[],2);
            temp = (idx(:,i)-1)*n+(1:n)';
            W(temp) = 1e100; 
        end
        
        %test self-tune sigma
%         sigma = dump(:,s);
%         dump = exp(-dump ./ (2.0 .* sigma .^ 2));

        dump = exp(-dump/(2.0*sigma^2));
        Gidx = repmat((1:n)',1,s);
        Gjdx = idx;
        W = sparse(Gidx(:),Gjdx(:),dump(:),n,r);
        %remove outlier
        W = W(kept_idx,:);
        
    elseif s <= 0 % default to dense matrix
        fprintf(fileid,'default to dense matrix\n');
        if strcmp(embed_method, 'landmark')
            [~,idx] = min(W,[],2);
        end
        W = exp(-W/(2.0*sigma^2));
        %remove outlier
        W = W(kept_idx,:);
    end
    fprintf(fileid,'done in %.2f seconds\n',toc);
end

%% compute laplacian

fprintf(fileid,'Computing Laplacian and diffusion map...\n');
tic;

d1 = sum(W, 2);
d2 = sum(W, 1);
d1 = max(d1, 1e-15);
d2 = max(d2, 1e-15);
D1 = sparse(1:no_kept,1:no_kept,d1.^(-0.5));
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
fprintf(fileid,'Done in %.2f seconds\n',toc);

%% cluster embeddings

fprintf(fileid,'Clustering result embeddings...\n');
tic;
if strcmp(embed_method, 'landmark')
    if strcmp(cluster_method, 'kmeans')
        V(:,1) = [];
        V = V ./ sqrt(sum(V .^2, 2));
        reps_labels = litekmeans(V, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
    elseif strcmp(cluster_method, 'discretize') 
        reps_labels = discretize(V);
    end
    label = zeros(n, 1);
    for i = 1:n
        label(i) = reps_labels(idx(i));
    end
    
elseif strcmp(embed_method, 'direct')
    if strcmp(cluster_method, 'kmeans')
        U(:,1) = [];
        U = U ./ sqrt(sum(U .^2, 2));
        label = litekmeans(U, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
    elseif strcmp(cluster_method, 'discretize')
        label = discretize(U);
    end
    
elseif strcmp(embed_method, 'coclustering')
    W = [U;V];
    if strcmp(cluster_method, 'kmeans')
        W(:,1) = [];
        W = W ./ sqrt(sum(W .^2, 2));
        all_label = litekmeans(W, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
    elseif strcmp(cluster_method, 'discretize')
        all_label = discretize(W);
    end
    label = all_label(1:n);
end
fprintf(fileid,'Done in %.2f seconds\n', toc);

end

