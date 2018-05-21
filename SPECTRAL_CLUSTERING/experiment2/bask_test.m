function bask_test(mat, affinity, seed, r, s, maxt, maxit)

% gaussian
fprintf('Processing %s data set\n', mat);
addpath(genpath('/srv/home/kpham/MATLAB/'));
% addpath('pickled/');
% addpath('deng cai/');
% addpath('dataset/');
% addpath('dataset/paper_data/');
load(mat);
nlabel = max(gnd);
rng(seed);
initIter = 10;
initRes = 1;

acc1=zeros(maxit,1);
acc2=zeros(maxit,1); 
acc3=zeros(maxit,maxt);
acc4=zeros(maxit,maxt);
acc5=zeros(maxit,maxt);
acc6 = zeros(maxit,1);
acc7 = zeros(maxit,1);
acc8 = zeros(maxit,1);
lsc_acc = zeros(maxit,1);
kasp_acc = zeros(maxit,1);

km_t = zeros(maxit,1);
lsc_t = zeros(maxit,1);
kasp_t = zeros(maxit,1);
original_t = zeros(maxit,1);
reps_t = zeros(maxit,1);
all_t = zeros(maxit,1);

if strcmp(affinity, 'cosine')
    fea = bsxfun(@rdivide, fea, sqrt(sum(fea.^2, 2)));
end

for i = 1:maxit
    %common representatives

    t0 = cputime;
    if strcmp(affinity, 'cosine')
        distance = 'cosine';
    else 
        distance = 'sqEuclidean';
    end
    [lb, reps, ~, VAR] = litekmeans(fea, r, 'Distance', distance,...
        'replicates', initRes, 'maxiter', initIter, 'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100, 'clustersample', 0.1);
    km_t(i) = cputime - t0;
    lbcount = hist(lb, 1:r);
    cluster_sigma = sqrt(VAR ./ lbcount);
    sigma = mean(cluster_sigma);
    
    % setting up tops
    fprintf('Iteration %d:\n', i);

    opts.r = r;
    opts.s = s;
    opts.reps = reps;
    opts.lbcount = lbcount;
    opts.sigma = sigma;
    
    %LSC
    t0 = cputime;res = lsc(fea, nlabel, opts); lsc_t(i) = cputime - t0;
    lsc_acc(i) = bestacc(res, gnd);
    fprintf('LSC: %f\n', lsc_acc(i));
    
    %KASP
    opts.sigma = sigma;
    opts.pre_label = lb;
    t0 = cputime;res = KASP(fea, nlabel, r, opts); kasp_t(i) = cputime - t0;
	kasp_acc(i) = bestacc(res, gnd);
    fprintf('KASP: %f\n', kasp_acc(i));

    % MY ALGORITHM
    [l1,l2,l3,l4,l5,l6,l7,l8,original_t(i),reps_t(i),all_t(i)] = bask_t(fea, nlabel, affinity, maxt, opts);
    
    acc1(i) = bestacc(l1, gnd);
    acc2(i) = bestacc(l2, gnd);
    fprintf('spectral clustering, original data: %f\n', acc1(i));
    fprintf('spectral clustering, representative: %f\n', acc2(i));
    for j = 1:maxt
        acc3(i,j) = bestacc(l3(:,j),gnd);
        acc4(i,j) = bestacc(l4(:,j),gnd);
        acc5(i,j) = bestacc(l5(:,j),gnd);
        fprintf('diffusion map, original data, step size = %d: %f\n', j*2, acc3(i,j));
        fprintf('diffusion map, representative, step size = %d: %f\n',j*2, acc4(i,j));
        fprintf('diffusion map, both, step size = %d: %f\n', j*2-1, acc5(i,j));
       
    end   
    clear opts;
    
    acc6(i) = bestacc(l6, gnd);
    acc7(i) = bestacc(l7, gnd);
    acc8(i) = bestacc(l8, gnd);
    fprintf('spectral clustering, dhillon intepretation: %f\n', acc6(i));
    fprintf('spectral clustering, njw, original: %f\n',acc7(i));
    fprintf('spectral clustering, njw, reps: %f\n', acc8(i));
     
end
% Average accuracy
fprintf('\nAverage accuracy:\n')
fprintf('LSC: %f\n', mean(lsc_acc));
fprintf('KASP: %f\n', mean(kasp_acc));
fprintf('spectral clustering, original data: %f\n', mean(acc1));
fprintf('spectral clustering, representative: %f\n', mean(acc2));
for j = 1:maxt
    fprintf('diffusion map, original data, step size = %d: %f\n', j*2, mean(acc3(:,j)));
    fprintf('diffusion map, representative, step size = %d: %f\n',j*2, mean(acc4(:,j)));
    fprintf('diffusion map, all, step size = %d: %f\n', j*2-1, mean(acc5(:,j)));
end
fprintf('spectral clustering, dhillon intepretation: %f\n', mean(acc6));
fprintf('spectral clustering, njw, original: %f\n',mean(acc7));
fprintf('spectral clustering, njw, reps: %f\n', mean(acc8));

save(strcat(mat, '_bask_result'), 'acc1', 'acc2','acc3','acc4','acc5', ...
    'acc6','acc7','acc8','lsc_acc', 'kasp_acc', ...
    'km_t','lsc_t','kasp_t','original_t','reps_t','all_t');
clear;
end

function acc = bestacc(label, gnd)
l = bestMap(gnd, label);
acc = sum((l - gnd) == 0) / size(gnd, 1);
end
