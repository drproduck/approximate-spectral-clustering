
addpath(genpath('home/drproduck/Documents/MATLAB/'));
load('usps.mat');
r = 500;
s = 5;
nlabel = max(gnd);
n = size(fea, 1);

[lb, reps, ~, VAR] = litekmeans(fea, r, 'replicates', 1, 'maxiter', 10, 'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100, 'clustersample', 0.1);
lbcount = hist(lb, 1:r);
cluster_sigma = sqrt(VAR ./ lbcount);
sigma = mean(cluster_sigma);


opts.r = r;
opts.s = s;
opts.reps = reps;
opts.lbcount = lbcount;
opts.sigma = sigma;

[U,S,V] = bask_eig(fea, nlabel, 'gaussian', opts);
U(:,1) = [];
U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));


fprintf('200, 20\n')
for i = 1:10
    [l,~,~,score] = litekmeans(U, nlabel, 'Distance', 'sqEuclidean', 'MaxIter', 200, 'Replicates',20);
    res = bestMap(gnd, l);
    sum(score)
    acc = sum(res == gnd) / n
end

fprintf('100, 10\n')
for i = 1:10
    [l,~,~,score] = litekmeans(U, nlabel, 'Distance', 'sqEuclidean', 'MaxIter', 100, 'Replicates',10);
    res = bestMap(gnd, l);
    sum(score)
    acc = sum(res == gnd) / n
end

fprintf('100, 10, cosine\n')
for i = 1:10
    [l,~,~,score] = litekmeans(U, nlabel, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates',10);
    res = bestMap(gnd, l);
    sum(score)
    acc = sum(res == gnd) / n
end

fprintf('cluster 10\n')
for i = 1:10
    [l,~,~,score] = litekmeans(U, nlabel, 'replicates', 1, 'maxiter', 100, 'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100, 'clustersample', 0.1);
    res = bestMap(gnd, l);
    sum(score)
    acc = sum(res == gnd) / n
end

fprintf('fast\n')
for i = 1:10
    [l,~,~,score] = litekmeans(U, nlabel, 'Distance', 'cosine', 'replicates', 1, 'maxiter', 10);
    res = bestMap(gnd, l);
    sum(score)
    acc = sum(res == gnd) / n
end


