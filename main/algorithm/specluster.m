function [labels, labels_out] = specluster(X, k, opts)

% Spectral clustering
%
% Input
%   X: n by d data matrix
%   k: number of clusters
%   opts: a structure array with the following fields
%       .clustering: 'NJW' (default) or 'diffusionMap'
%       .steps: number of steps in diffusion map (default=0, Ncut)
%       .affinity: 'Gaussian'  or 'cosine'(default)  or 'correlation'
%       .sigmaValue: radius or # nearest neighbors (default=7), or power (default=1) depending on sigmaMethod used below
%       .sigmaMethod: a string, one of the following
%                'direct': in this case sigmaValue refers to the sigma in the gaussian kernel
%                'meanDist'(default): mean distance of points to their sigmaValue nearest neighbors
%                'medianDist': median distance of points to their sigmaValue nearest neighbors
%                'selfTuning': sigmaValue = # nearest neighbors
%       .normalizeL: whether or not to normalize the graph Laplacian. Default = true
%       .outlierFraction: a number between 0 (default) and 1
% Ouput
%   labels: clusters labels found by the function

%% set parameters

if nargin<3
    opts = struct();
end

if ~isfield(opts, 'clustering')
    opts.clustering = 'NJW';
end

if strcmp(opts.clustering, 'diffusionMap') && ~isfield(opts,'steps')
    opts.steps = 0;
end


if ~isfield(opts, 'affinity')
    opts.affinity = 'cosine';
end


switch(opts.affinity)
    
    case 'Gaussian'

        if ~isfield(opts, 'sigmaMethod')
            opts.sigmaMethod = 'meanDist';
        end
        
        if ~isfield(opts, 'sigmaValue')
            opts.sigmaValue = 7;
        end

    case 'cosine'
        
         if ~isfield(opts, 'sigmaMethod')
            opts.sigmaMethod = 'power';
         end
        
        if ~isfield(opts, 'sigmaValue')
            opts.sigmaValue = 1;
        end
        
end

if ~isfield(opts, 'normalizeL')
    opts.normalizeL = true;
end


if ~isfield(opts, 'outlierFraction')
    opts.outlierFraction = 0;
end

%% calculating affinities

[n, d] = size(X); % number of data points

norms2 = sum(X.*X,2);

switch opts.affinity
    
    case 'Gaussian'
        
        dists2 = -(2*X)*X';
        dists2 = bsxfun(@plus, dists2, norms2);
        dists2 = bsxfun(@plus, dists2, norms2');
        %dists2 = repmat(norms2,1,n) + repmat(norms2',n,1) - (2*X)*X';
        dists2_sort = sort(dists2, 2, 'ascend');
        
        switch opts.sigmaMethod
            case 'sigma'
                sigma2 = 2*opts.sigmaValue^2;
            case 'meanDist'
                sigma2 = 2*mean(sqrt(dists2_sort(:, opts.sigmaValue)))^2;
            case 'medianDist'
                sigma2 = 2*median(sqrt(dists2_sort(:, opts.sigmaValue)))^2;
            case 'selfTuning'
                sigma2 = 2*sqrt(dists2_sort(:,opts.sigmaValue))*sqrt(dists2_sort(:,opts.sigmaValue))';
        end
        
        W = exp(-dists2 ./ sigma2);

    case 'cosine'
        
        %X_norm = X./repmat(sqrt(norms2),1,size(X,2));
        X_norm = bsxfun(@rdivide,X,sqrt(norms2));
        W = X_norm*X_norm';
        
        if opts.sigmaValue > 1
            W = W.^opts.sigmaValue;
        elseif opts.sigmaValue<0
            W = W./(2-W);
        end
        
    case 'correlation'
        
        W = corrcoef(X');
 
end

W(W<1e-16) = 0;
W(1:n+1:end) = 0;
%figure; imagesc(W); title('weight matrix')
        
%% remove outliers by using degrees
%inliers = 1:n;

degrees = sum(W,2);
outliers = (degrees<1e-8);
n_outliers = sum(outliers);

if n_outliers >= ceil(n*opts.outlierFraction)
    
    inliers = ~outliers;
    
    W = W(inliers, inliers);
    degrees = degrees(inliers);
    
else
    
    n_outliers = ceil(n*opts.outlierFraction);
    
    [~, inds_sort_degrees] = sort(degrees,'ascend');
    %figure; plot(sorted_degrees, '.', 'markersize',12)
    
    inliers = inds_sort_degrees(n_outliers+1:end);
    outliers = inds_sort_degrees(1:n_outliers);
    
    W = W(inliers, inliers);
    degrees = sum(W,2);
    
end

n = n - n_outliers;

%% normalization

dvec_inv = 1./sqrt(degrees);
%W_tilde = repmat(dvec_inv, 1, n).*W.*repmat(dvec_inv', n, 1);
%W_tilde = (dvec_inv*dvec_inv').*W;
W_tilde = bsxfun(@times, W, dvec_inv);
W_tilde = bsxfun(@times, W_tilde, dvec_inv');
W_tilde = (W_tilde+W_tilde')/2;

%% calculate eigenvectors
[V,lambdas] = eigs(W_tilde, k, 'LM', struct('issym', true, 'isreal', true));

switch opts.clustering
    case 'diffusionMap'
        %V = repmat(dvec_inv, 1, k-1).*V(:,2:k);
        V = bsxfun(@times, V(:,2:k), dvec_inv);
        if opts.steps>0
            lambdas = diag(lambdas(2:k,2:k));
            %V = repmat(lambdas'.^opts.steps, n, 1).*V;
            V = bsxfun(@times, V, lambdas'.^opts.steps);
        end
        %V = V./repmat(sqrt(sum(V.*V,2)), 1, k-1);
        V = bsxfun(@rdivide, V, sqrt(sum(V.*V,2)));
    case 'NJW'
        %V = V./repmat(sqrt(sum(V.*V,2)), 1, k);
        V = bsxfun(@rdivide, V, sqrt(sum(V.*V,2)));
end

%% kmeans clustering
if n_outliers > 0
    
    labels = zeros(n+n_outliers,1);
    labels(inliers) = kmeans(V, k, 'replicates', 10);
    
    %figure; gcplot(V, labels(inliers)); title('clusters found by kmeans in eigenvector space')

    mdl = fitcknn(X(inliers,:), labels(inliers), 'NumNeighbors', 3);
    labels_out = predict(mdl, X(outliers,:));
    
    %centers = zeros(k,d);
    %for i = 1:k
    %    centers(i,:) = mean(X(labels==i,:),1);
    %    centers(i,:) = centers(i,:)/norm(centers(i,:));
    %end
    
    %[~, labels_out] = max(X(outliers,:)*centers',[],2);
    
else
    
    labels = kmeans(V, k, 'replicates', 10);
    %figure; gcplot(V,labels); title('clusters found by kmeans in eigenvector space')
    labels_out = [];

end
