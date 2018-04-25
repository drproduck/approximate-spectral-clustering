
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

for r = R(1):R(2):R(3)
    for it = 1:maxit
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
        fprintf('# landmarks %d:\n', r*100);
        fprintf('Iteration %d:\n', it);

        opts.r = r * 100;
        opts.s = s;
        opts.reps = reps;
        opts.lbcount = lbcount;
        opts.sigma = sigma;

        %LSC
        t0 = cputime;res = lsc(fea, nlabel, opts); lsc_t(r,it) = cputime - t0;
        lsc_acc(r,it) = bestacc(res, gnd);
        fprintf('LSC: %f\n', lsc_acc(r,it));

        %KASP
        opts.pre_label = lb;
        t0 = cputime;res = kasp(fea, nlabel, r * 100, opts); kasp_t(r,it) = cputime - t0;
        kasp_acc(r,it) = bestacc(res, gnd);
        fprintf('KASP: %f\n', kasp_acc(r,it));

        % MY ALGORITHM
        [l3,l4,dhillon_acc,cspec_acc,t3,t4,dhillon_t,cspec_t] = bask_t_3(fea, nlabel, affinity, maxt, opts);
           
        ti3(r,it,:) = t3;
        ti4(r,it,:) = t4;
        tid(r,it) = dhillon_t;
        tic(r,it) = cspec_t;
            
        for t = 1:maxt
            acc3(r,it,t) = bestacc(l3(:,t),gnd);
            acc4(r,it,t) = bestacc(l4(:,t),gnd);
            fprintf('lbdm x %d: %f\n', t*2, acc3(r,it,t));
            fprintf('lbdm y %d: %f\n',t*2, acc4(r,it,t));
        end   
        
        acc_d(r,it) = bestacc(dhillon_acc, gnd);
        acc_c(r,it) = bestacc(cspec_acc, gnd);
        fprintf('dhillon: %f\n', acc_d(r,it));
        fprintf('cspec: %f\n',acc_c(r,it));     
    end
    
    % Average accuracy
    fprintf('\nAverage accuracy for r = %d:\n', r*100)
    fprintf('LSC: %f\n', mean(lsc_acc(r,:)));
    fprintf('KASP: %f\n', mean(kasp_acc(r,:)));
    for t = 1:maxt
        fprintf('lbdm x %d: %f\n', t*2, mean(acc3(r,:,t)));
        fprintf('lbdm y %d: %f\n',t*2, mean(acc4(r,:,t)));
    end
    fprintf('dhillon: %f\n', mean(acc_d(r,:)));
    fprintf('cspec: %f\n',mean(acc_c(r,:)));
end

save(strcat(mat, '_m_sens_result'), 'acc3','acc4', ...
    'acc_d', 'acc_c','lsc_acc', 'kasp_acc', ...
    'km_t','lsc_t','kasp_t','ti3','ti4','tid','tic');
clear;
end

function acc = bestacc(label, gnd)
l = bestMap(gnd, label);
acc = sum((l - gnd) == 0) / size(gnd, 1);
end
