function [labels, outliers] = ssc_cosine(X, k, opts)

% Scalable spectral clustering with cosine similarity
%
% Input
%   X: n by d data matrix
%   k: number of clusters
%   opts: a structure array with the following fields
%     .isRowNormalized: whether the data has been normalized to have unit
%         length
%     .t: a variable indicating the clustering method to be used:
%         t=-1 NJW  
%         t=0: NCut (default) 
%         t>=1: DM with t time steps
%     .alpha: a number between 0 (default) and 1 representing the
%         fraction of outliers to be removed
%     .classify_outliers: logical variable indicating whether to 
%         classify outliers back to the data. Default = true.
%       
% Ouput
%   labels: group labels {1,...,k) or {0,1,...,k} found by algorithm: 
%           (1) when opts.classify_outliers = true, outliers will be 
%               classified into the clusters and thus receive positive labels;
%           (2) when opts.classify_outliers = false, no classification
%               will be done and outliers have zero labels 
%   outliers: logical variable indicating whether each point is an outlier.
%
% Reference: 
%   Scalable spectral clustering with cosine similarity, G. Chen, Proc. of
%   International Conference on Pattern Recognition (ICPR), Beijing, China,
%   August 2018
%
% @Guangliang Chen (2018)

%% set parameters

if nargin<3
    opts = struct( );
end

if ~isfield(opts,'isRowNormalized')
    opts.isRowNormalized = false;
end

if ~isfield(opts, 't')
    opts.t = 0; % default = NCut
end

if ~isfield(opts, 'alpha')
    opts.alpha = 0.01; % default = 1% outliers removed
end

if ~isfield(opts, 'classify_outliers')
    opts.classify_outliers = true; % default = 1% outliers removed
end

n = size(X,1); 
outliers = false(1,n); % all data points are normal points initially

NUMERICAL_ZERO = 1e-16;

%% calculating degrees

if opts.isRowNormalized 
    X_norm = X;
else
    X_norm = bsxfun(@rdivide, X, sqrt(sum(X.*X,2)));
end

degrees = X_norm*sum(X_norm,1)' - 1;

%% remove outliers according to degrees

% points with zero degrees are automatically treated as outliers
deg0_outliers = (degrees<NUMERICAL_ZERO);
n_out_0 = sum(deg0_outliers); 

outliers(deg0_outliers) = true; % 0 encodes outliers

n_out = round(n*opts.alpha);
if n_out > n_out_0
    
    degrees(deg0_outliers) = n+1;
    for i = 1:n_out-n_out_0
        [~,ind_out_i] = min(degrees);
        outliers(ind_out_i) = true;
        degrees(ind_out_i) = n+1;
    end
    
    %[~, inds_sort_degrees] = sort(degrees,'ascend');
    %inliers(inds_sort_degrees(1:n_out)) = false; 
    
end

inliers = ~outliers;

X_norm_in = X_norm(inliers,:);
degs_in = degrees(inliers);

% recompute the degrees
%degs_in = X_norm_in*sum(X_norm_in,1)' - 1;

%% normalizing X

dvec_inv_in = 1./sqrt(degs_in);
X_in_tilde = bsxfun(@times, X_norm_in, dvec_inv_in);

%% calculating eigenvectors

[U_in,S] = svds(X_in_tilde, k); % enough for NJW clustering

if opts.t >= 0 % NCut or DM
    U_in = bsxfun(@times, U_in(:,2:k), dvec_inv_in);
end

if opts.t > 0 % DM with t time steps    
    lambdas = diag(S(2:k,2:k))'.^2 - mean(dvec_inv_in);
    U_in = bsxfun(@times, U_in, lambdas.^opts.t);
end

% L_2 row normalization
Y = bsxfun(@rdivide, U_in, sqrt(sum(U_in.^2,2)));

%% kmeans clustering

labels = zeros(1,n);

if exist('kmeans','file')
    labels(inliers) = kmeans(Y, k, 'replicates', 10);
elseif exist('litekmeans','file')
    labels(inliers) = litekmeans(Y, k, 'replicates', 10); 
else 
    error('kmeans and litekmeans not found!')
end

%% classify outliers back to data
if opts.classify_outliers && any(outliers)
    
    % nearest centroid classification
    centers = zeros(k,size(X,2));
    for i = 1:k
        cls_i = (labels==i);
        centers(i,:) = mean(X_norm(cls_i,:),1);
        %centers(i,:) = centers(i,:)/norm(centers(i,:));
    end
    
    [~, labels(outliers)] = max(X_norm(outliers,:)*centers',[],2);
    
    % kNN classification
    %mdl = fitcknn(X_norm(inliers,:), labels(inliers), 'NumNeighbors', 3, 'Distance', 'cosine');
    %labels(outliers) = predict(mdl, X_norm(outliers,:));
    
end
    
end
