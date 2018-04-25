
function s_sens_test(mat, affinity, seed, r, S, maxt, maxit)
% fix m (landmark), vary s (sparse)

fprintf('Processing %s data set\n', mat);
addpath(genpath('/home/drproduck/Documents/MATLAB/'));
load(mat);
nlabel = max(gnd);
rng(seed);
initIter = 10;
initRes = 1;

m = (S(3) - S(1)) / S(2) + 1;
acc3=zeros(m, maxit,maxt);
acc4=zeros(m, maxit,maxt);
acc_d = zeros(m, maxit,1);
acc_c = zeros(m, maxit,1);
lsc_acc = zeros(m, maxit,1);
kasp_acc = zeros(m, maxit,1);

km_t = zeros(m, maxit,1);
lsc_t = zeros(m, maxit,1);
kasp_t = zeros(m, maxit,1);
ti3 = zeros(m,maxit,maxt);
ti4 = zeros(m,maxit,maxt);
tid = zeros(m,maxit,1);
tic = zeros(m,maxit,1);


if strcmp(affinity, 'cosine')
    fea = bsxfun(@rdivide, fea, sqrt(sum(fea.^2, 2)));
end

for s = S(1):S(2):S(3)
    for it = 1:maxit
        %common representatives

        t0 = cputime;
        if strcmp(affinity, 'cosine')
            distance = 'cosine';
        else 
            distance = 'sqEuclidean';
        end
        [lb, reps, ~, VAR] = litekmeans(fea, r, 'Distance', distance, 'replicates', initRes, 'maxiter', initIter, 'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100, 'clustersample', 0.1);
        km_t(s, it) = cputime - t0;
        lbcount = hist(lb, 1:r);
        cluster_sigma = sqrt(VAR ./ lbcount);
        sigma = mean(cluster_sigma);

        % setting up opts
        fprintf('# nearest neighbors %d:\n', s);
        fprintf('Iteration %d:\n', it);

        opts.r = r;
        opts.s = s;
        opts.reps = reps;
        opts.lbcount = lbcount;
        opts.sigma = sigma;

        %LSC
        t0 = cputime;res = lsc(fea, nlabel, opts); lsc_t(s,it) = cputime - t0;
        lsc_acc(s,it) = bestacc(res, gnd);
        fprintf('LSC: %f\n', lsc_acc(s,it));

        %KASP
        opts.pre_label = lb;
        t0 = cputime;res = kasp(fea, nlabel, r, opts); kasp_t(s,it) = cputime - t0;
        kasp_acc(s,it) = bestacc(res, gnd);
        fprintf('KASP: %f\n', kasp_acc(s,it));

        % MY ALGORITHM
        [l3,l4,dhillon_acc,cspec_acc,t3,t4,dhillon_t,cspec_t] = bask_t_3(fea, nlabel, affinity, maxt, opts);
           
        ti3(s,it,:) = t3;
        ti4(s,it,:) = t4;
        tid(s,it) = dhillon_t;
        tic(s,it) = cspec_t;
            
        for t = 1:maxt
            acc3(s,it,t) = bestacc(l3(:,t),gnd);
            acc4(s,it,t) = bestacc(l4(:,t),gnd);
            fprintf('lbdm x %d: %f\n', t*2, acc3(s,it,t));
            fprintf('lbdm y %d: %f\n',t*2, acc4(s,it,t));
        end   
        
        acc_d(s,it) = bestacc(dhillon_acc, gnd);
        acc_c(s,it) = bestacc(cspec_acc, gnd);
        fprintf('dhillon: %f\n', acc_d(s,it));
        fprintf('cspec: %f\n',acc_c(s,it));     
    end
    
    % Average accuracy
    fprintf('\nAverage accuracy for s = %d:\n', s)
    fprintf('LSC: %f\n', mean(lsc_acc(s,:)));
    fprintf('KASP: %f\n', mean(kasp_acc(s,:)));
    for t = 1:maxt
        fprintf('lbdm x %d: %f\n', t*2, mean(acc3(s,:,t)));
        fprintf('lbdm y %d: %f\n',t*2, mean(acc4(s,:,t)));
    end
    fprintf('dhillon: %f\n', mean(acc_d(s,:)));
    fprintf('cspec: %f\n',mean(acc_c(s,:)));
end

save(strcat(mat, '_s_sens_result'), 'acc3','acc4', ...
    'acc_d', 'acc_c','lsc_acc', 'kasp_acc', ...
    'km_t','lsc_t','kasp_t','ti3','ti4','tid','tic');
clear;
end

function acc = bestacc(label, gnd)
l = bestMap(gnd, label);
acc = sum((l - gnd) == 0) / size(gnd, 1);
end
