function wrapper_careful_test(mat, seed, r, sparse, maxit)

% gaussian
fprintf('Processing %s data set\n', mat);
addpath('pickled/');
addpath('deng cai/');
addpath('dataset/');
addpath('dataset/paper_data/');
load(mat);
nlabel = max(gnd);
result_file_name = strcat(mat, '_t=3_test_result');
file = fopen(result_file_name, 'w');
rng(seed);
initIter = 10;
initRes = 1;

t = zeros(maxit, 4);
s = zeros(maxit, 4);

for i = 1:maxit
    %common representatives
    
    tic;
%     [lb, reps, ~, VAR] = litekmeans(fea, r, 'MaxIter', initIter, 'Replicates', initRes);
    [lb, reps, ~, VAR] = litekmeans(fea, r, 'replicates', initRes, 'maxiter', initIter, 'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100, 'clustersample', 0.1);
    add = toc
    lbcount = hist(lb, 1:r);
    cluster_sigma = sqrt(VAR ./ lbcount);
    sigma = mean(cluster_sigma);
    
    %LSC

    fprintf('LSC run %d:\n', i);
    fprintf(file,'LSC run %d:\n', i);
    opts.r = sparse;
    opts.p = r;
    opts.reps = reps;
    opts.sigma = sigma;
    tic;res = LSC(fea, nlabel, opts); t(i,3) = toc + add;
    res = bestMap(gnd,res);
    s(i,3) = sum((res - gnd) == 0) / size(gnd, 1);
    fprintf('ac = %f, time = %f\n', s(i,3), t(i,3));
    fprintf(file, 'ac = %f, time = %f\n', s(i,3), t(i,3));
    clear opts;
    
    %KASP
    fprintf('KASP run %d:\n', i);
    fprintf(file,'KASP run %d:\n', i);
    opts.reps = reps;
    opts.sigma = sigma;
    opts.pre_label = lb;
    tic;res = KASP(fea, nlabel, r, opts); t(i,4) = toc + add;
    res = bestMap(gnd,res);
    s(i,4) = sum((res - gnd) == 0) / size(gnd, 1);
    fprintf('ac = %f, time = %f\n', s(i,4), t(i,4));
    fprintf(file, 'ac = %f, time = %f\n', s(i,4), t(i,4));
    clear opts;
    
    %BIAS
    
    fprintf('BIAS run %d:\n', i);
    fprintf(file,'BIAS run %d:\n', i);
    opts.r = r;
    opts.sparse = sparse;
    opts.reps = reps;
    opts.lbcount = lbcount;
    opts.sigma = sigma;
    opts.t = 3;
    tic;[label, ~] = BiAS(fea, nlabel, 'gaussian', opts); t(i,1) = toc + add;
    label = bestMap(gnd, label);
    s(i,1) = sum((label - gnd) == 0) / size(gnd, 1);
    fprintf('ac = %f, time = %f\n', s(i,1), t(i,1));
    fprintf(file,'ac = %f, time = %f\n', s(i,1), t(i,1));
    clear opts;
    
    %BIASk
    
    fprintf('BIASk run %d:\n', i);
    fprintf(file,'BIASk run %d:\n', i);
    opts.r = r;
    opts.sparse = sparse;
    opts.kasp = 1;
    opts.reps = reps;
    opts.lbcount = lbcount;
    opts.sigma = sigma;
    opts.t = 3;
    tic;[label, ~] = BiAS(fea, nlabel, 'gaussian', opts); t(i,2) = toc + add;
    label = bestMap(gnd, label);
    s(i,2) = sum((label - gnd) == 0) / size(gnd, 1);
    fprintf('ac = %f, time = %f\n', s(i,2), t(i,2));
    fprintf(file,'ac = %f, time = %f\n', s(i,2), t(i,2));
    clear opts;
     
end

mt = mean(t);
ms = mean(s);
stds = std(s);

fprintf('BIAS average: mean ac = %f, std ac = %f, time = %f\n', ms(1), stds(1), mt(1));
fprintf(file, 'BIAS average: mean ac = %f, std ac = %f, time = %f\n', ms(1), stds(1), mt(1));

fprintf('BIASk average: mean ac = %f, std ac = %f, time = %f\n', ms(2), stds(2), mt(2));
fprintf(file, 'BIASk average: mean ac = %f, std ac = %f, time = %f\n', ms(2), stds(2), mt(2));

fprintf('LSC average: mean ac = %f, std ac = %f, time = %f\n', ms(3), stds(3), mt(3));
fprintf(file, 'LSC average: mean ac = %f, std ac = %f, time = %f\n', ms(3), stds(3), mt(3));
 
fprintf('KASP average: mean ac = %f, std ac = %f, time = %f\n', ms(4), stds(4), mt(4));
fprintf(file, 'KASP average: mean ac = %f, std ac = %f, time = %f\n', ms(4), stds(4), mt(4));

fclose(file);
save(strcat(mat, '_kmeans_t=3_stats'), 's', 't');
clear;
end
