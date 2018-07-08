addpath(genpath('home/drproduck/Documents/MATLAB/'));
load('news.mat');
r = 500;
s = 5;
nlabel = max(gnd);
n = size(fea, 1);
fea = bsxfun(@rdivide, fea, sqrt(sum(fea.^2, 2)));
[lb, reps, ~, VAR] = litekmeans(fea, r, 'Distance', 'cosine', 'replicates', 1, 'maxiter', 10, 'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100, 'clustersample', 0.1);
lbcount = hist(lb, 1:r);
cluster_sigma = sqrt(VAR ./ lbcount);
sigma = mean(cluster_sigma);

opts.r = r;
opts.s = s;
opts.reps = reps;
opts.lbcount = lbcount;
opts.sigma = sigma;

[U,S,V,idx] = bask_eig(fea, nlabel, 'cosine', opts);
V(:,1) = [];

V = bsxfun(@rdivide, V, sqrt(sum(V.^2, 2)));
reps_label = litekmeans(V, nlabel, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates',10);
l8 = zeros(n, 1);
for i = 1:n
    % give labels of r-nearest reps, pick label with the most occurence. 
    can_reps_label = reps_label(idx(i,:));
    [~, m] = max(hist(can_reps_label, 1:nlabel));
    l8(i) = m;
end

l8 = bestMap(gnd, l8);
acc = sum(l8 == gnd) / size(gnd, 1)