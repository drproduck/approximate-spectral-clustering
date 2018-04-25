function dhillon_test(mat, affinity, seed, r, s, maxit)

fprintf('Processing %s data set\n', mat);
addpath(genpath('/home/drproduck/Documents/MATLAB/'));
load(mat);
nlabel = max(gnd);
rng(seed);
n = size(fea, 1);

if strcmp(affinity, 'cosine')
    fea = bsxfun(@rdivide, fea, sqrt(sum(fea.^2, 2)));
end
% cosine => cosine, gaussian => sqEuclidean
if strcmp(affinity, 'cosine')
    distance = affinity;
else
    distance = 'sqEuclidean';
end

acc1 = zeros(n,1);
acc2 = zeros(n,1);

for i = 1:maxit
    [lb, reps, ~, VAR] = litekmeans(fea, r, 'Distance', distance, 'replicates', 1, 'maxiter', 10, 'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100, 'clustersample', 0.1);
    lbcount = hist(lb, 1:r);
    cluster_sigma = sqrt(VAR ./ lbcount);
    sigma = mean(cluster_sigma);

    opts.r = r;
    opts.s = s;
    opts.reps = reps;
    opts.lbcount = lbcount;
    opts.sigma = sigma;

    [UN,SN,VN,idx,D1,D2] = bask_eig(fea, nlabel, affinity, opts);
    % original dhillon
    U = D1 * UN;
    V = D2 * VN;
    W = [U;V];
    W(:,1) = [];
    W = bsxfun(@rdivide, W, sqrt(sum(W.^2, 2)));
    label = litekmeans(W, nlabel, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates',10);
    label = label(1:n);
    label = bestMap(gnd, label);
    acc1(i) = sum(label == gnd) / size(gnd, 1);
    fprintf('original dhillon: %f\n', acc1(i));

    % dhillon diffusion map
    U = D1 * UN * SN;
    V = D2 * VN * SN;
    W = [U;V];
    W(:,1) = [];
    W = bsxfun(@rdivide, W, sqrt(sum(W.^2, 2)));
    label = litekmeans(W, nlabel, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates',10);
    label = label(1:n);
    label = bestMap(gnd, label);
    acc2(i) = sum(label == gnd) / size(gnd, 1);
    fprintf('dhillon diffusion: %f\n', acc2(i));
end

fprintf('\nAverage accuracy\n');
fprintf('original dhillon: %f\n', mean(acc1));
fprintf('dhillon diffusion: %f\n', mean(acc2));
save(strcat(mat, '_dhillon_test_result'), 'acc1', 'acc2');

end