clear;
maxt = 1;
maxit = 20;
seed = 99999;
R =  [1,1,10];
s = 5;
m_sensitivity_test('letter', 'gaussian', seed, R, s, maxt, maxit)
m_sensitivity_test('mnist', 'gaussian', seed, R, s, maxt,maxit)
m_sensitivity_test('usps', 'gaussian',seed, R, s, maxt,maxit)
m_sensitivity_test('protein', 'gaussian',seed, R, s, maxt,maxit)
% s_sens_test('pend', 'gaussian',seed, R, s, maxt,maxit)
% s_sens_test('shuttle', 'gaussian',seed, R, s, maxt,maxit)
% s_sens_test('musk_1', 'gaussian',seed, R, s, maxt,maxit)



function m_sensitivity_test(mat, affinity, seed, R, s, maxt, maxit)
% fix s (sparse), vary m (landmark)

fprintf('\nProcessing %s data set\n', mat);
addpath(genpath('/home/drproduck/Documents/MATLAB/'));
load(mat);
nlabel = max(gnd);
rng(seed);
initIter = 10;
initRes = 1;

m = (R(3) - R(1)) / R(2) + 1;
lbdm1_acc_all=zeros(m, maxit,maxt);

km_t = zeros(m,maxit,1);
lbdm1_t_all = zeros(m,maxit,1);



if strcmp(affinity, 'cosine')
    fea = bsxfun(@rdivide, fea, sqrt(sum(fea.^2, 2)));
end

fprintf('\n\n')
for r = R(1):R(2):R(3)
    for it = 1:maxit
        fprintf('Iteration %d:\n', it);
        %common representatives

        t0 = cputime;
        if strcmp(affinity, 'cosine')
            distance = 'cosine';
        else 
            distance = 'sqEuclidean';
        end
        [lb, reps, ~, VAR] = litekmeans(fea, r * 100, 'Distance', distance, 'replicates', initRes, 'maxiter', initIter, 'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100, 'clustersample', 0.1);
        km_t(r, it) = cputime - t0;
        lbcount = hist(lb, 1:r*100);
        cluster_sigma = sqrt(VAR ./ lbcount);
        sigma = mean(cluster_sigma);

        % setting up opts

        opts.r = r * 100;
        opts.s = s;
        opts.reps = reps;
        opts.lbcount = lbcount;
        opts.sigma = sigma;

        % MY ALGORITHM
        [lbdm1_acc, lbdm1_t] = bask_for_lbdm1(fea, nlabel, affinity, maxt, opts);
           
        lbdm1_t_all(r,it,:) = lbdm1_t;
            
        for t = 1:maxt
            lbdm1_acc_all(r,it,t) = bestacc(lbdm1_acc(:,t),gnd);
            fprintf('lbdm1 %d: %f\n', t*2-1, lbdm1_acc_all(r,it,t));
        end   
        
    end
    
    % Average accuracy
    fprintf('\nAverage accuracy for r = %d:\n', r*100)
    
    for t = 1:maxt
        fprintf('lbdm1 %d: %f\n', t*2-1, mean(lbdm1_acc_all(r,:,t))); 
    end
end

save(strcat(mat, '_m_sens_for_lbdm1'), 'lbdm1_acc_all', ...
    'km_t','lbdm1_t_all');
clear;
end

function acc = bestacc(label, gnd)
l = bestMap(gnd, label);
acc = sum((l - gnd) == 0) / size(gnd, 1);
end

