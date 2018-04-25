clear;
maxt = 1;
maxit = 20;
seed = 99999;
S = [1,1,10];
r = 500;

m_sensitivity_test('letter', 'gaussian', seed, r, S, maxt, maxit)
m_sensitivity_test('mnist', 'gaussian', seed, r, S, maxt,maxit)
m_sensitivity_test('usps', 'gaussian',seed, r, S, maxt,maxit)
m_sensitivity_test('protein', 'gaussian',seed, r, S, maxt,maxit)
% s_sens_test('pend', 'gaussian',seed, r, S, maxt,maxit)
% s_sens_test('shuttle', 'gaussian',seed, r, S, maxt,maxit)
% s_sens_test('musk_1', 'gaussian',seed, r, S, maxt,maxit)



function m_sensitivity_test(mat, affinity, seed, r, S, maxt, maxit)
% fix r (landmark), vary s( sparsity)

fprintf('\nProcessing %s data set\n', mat);
addpath(genpath('/home/drproduck/Documents/MATLAB/'));
load(mat);
nlabel = max(gnd);
rng(seed);
initIter = 10;
initRes = 1;

m = (S(3) - S(1)) / S(2) + 1;
lbdm1_acc_all=zeros(m, maxit,maxt);

km_t = zeros(m,maxit,1);
lbdm1_t_all = zeros(m,maxit,1);


if strcmp(affinity, 'cosine')
    fea = bsxfun(@rdivide, fea, sqrt(sum(fea.^2, 2)));
end

for s = S(1):S(2):S(3)
    fprintf('\n\n')
    for it = 1:maxit
        fprintf('Iteration %d:\n', it);
        %common representatives

        t0 = cputime;
        if strcmp(affinity, 'cosine')
            distance = 'cosine';
        else 
            distance = 'sqEuclidean';
        end
        fprintf('Getting %d landmarks\n', r)
        [lb, reps, ~, VAR] = litekmeans(fea, r, 'Distance', distance, 'replicates', initRes, 'maxiter', initIter, 'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100, 'clustersample', 0.1);
        km_t(s, it) = cputime - t0;
        lbcount = hist(lb, 1:r);
        cluster_sigma = sqrt(VAR ./ lbcount);
        sigma = mean(cluster_sigma);

        % setting up opts

        opts.r = r;
        opts.s = s;
        opts.reps = reps;
        opts.lbcount = lbcount;
        opts.sigma = sigma;

        % MY ALGORITHM
        [lbdm1_acc, lbdm1_t] = bask_for_lbdm1(fea, nlabel, affinity, maxt, opts);
           
        lbdm1_t_all(s,it,:) = lbdm1_t;
            
        for t = 1:maxt
            lbdm1_acc_all(s,it,t) = bestacc(lbdm1_acc(:,t),gnd);
            fprintf('lbdm1 %d: %f\n', t*2-1, lbdm1_acc_all(s,it,t));
        end   
        
    end
    
    % Average accuracy
    fprintf('\nAverage accuracy for s = %d:\n', s)
    
    for t = 1:maxt
        fprintf('lbdm1 %d: %f\n', t*2-1, mean(lbdm1_acc_all(s,:,t))); 
    end
end

save(strcat(mat, '_s_sens_for_lbdm1'), 'lbdm1_acc_all', ...
    'km_t','lbdm1_t_all');
clear;
end

function acc = bestacc(label, gnd)
l = bestMap(gnd, label);
acc = sum((l - gnd) == 0) / size(gnd, 1);
end

