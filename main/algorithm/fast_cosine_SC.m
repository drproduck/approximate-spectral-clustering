function [label,kept_idx] = fast_cosine_SC(fea, k, t, varargin)

%affinity_type:
                %doc_term
                %point

%input parser
p = inputParser;
addParameter(p,'remove_low',1);
addParameter(p,'remove_high',1);
addParameter(p,'embed_method','direct');
addParameter(p,'cluster_method','kmeans');
addParameter(p,'affinity_type','point');
parse(p,varargin{:});

remove_low = p.Results.remove_low;
remove_high = p.Results.remove_high;
embed_method = p.Results.embed_method;
cluster_method = p.Results.cluster_method;
affinity_type = p.Results.affinity_type;

[n,m] = size(fea);
fprintf('quickly finding row-sums...\n')
fea = fea ./ sqrt(sum(fea.^2, 2));
d = fea * (fea' * ones(n,1));
if (remove_low > 0 && remove_low < 1) || (remove_high > 0 && remove_high < 1)
    [val, amax] = sort(d, 'descend');
    plot(1:n,val);
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
    remove_idx = amax([1:high-1,low+1:n]);
    no_kept = size(kept_idx,1);
    plot(1:n,val);
    fprintf('remove %.0f%% of dataset...\n',(n - no_kept) / n *100);
	fea = fea(kept_idx,:);
    d = d(kept_idx,:);
else
    no_kept = n;
    kept_idx = [1:n];
end
fprintf('forming Laplacian...\n')
if strcmp(affinity_type, 'point')
    D = sparse(1:no_kept,1:no_kept,d.^(-0.5));
    [u,~,~] = svds(D * fea, k);
    u = D * u;
elseif strcmp(affinity_type,'doc_term')
    d1 = sum(fea, 2) .^ (-0.5);
    d2 = sum(fea, 1) .^ (-0.5);
    D1 = sparse(1:no_kept,1:no_kept, d1);
    D2 = sparse(1:m,1:m, d2);
    A = D1 * fea * D2;
    [u,s,v] = svds(A, k);
    if t == 0
        u = D1 * u;
        v = D2 * v;
    else
    u = D1 * u * s.^t;
    v = D2 * v * s.^t;
    end
    
end
fprintf('clustering result embeddings...\n')
if strcmp(embed_method, 'direct')
    if strcmp(cluster_method, 'kmeans')
        u(:,1) = [];
        u = u ./ sqrt(sum(u.^2, 2));
        label = litekmeans(u, k, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates',10);
    elseif strcmp(cluster_method, 'discretize')
        label = discretize(u);
    end
elseif strcmp(embed_method, 'coclustering') && ~strcmp(affinity_type, 'point')
    w = [u;v];
    w(:,1) = [];
    w = w ./ sqrt(sum(w.^2,2));
    all_label = litekmeans(w, k, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates', 10);
    label = all_label(1:no_kept);
end

end