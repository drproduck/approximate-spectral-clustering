function bask_test(mat, affinity, seed, r, s, maxt, maxit)
% r: number of representatives
% s: number of nearest neighbors landmarks to keep

% gaussian

disp('preparing variables...')

fprintf('Processing %s data set\n', mat);
mode = 'random';
load(mat, 'fea','gnd');
nlabel = max(gnd);
n = length(gnd);
rng(seed);
initIter = 10;
initRes = 1;

lsca=zeros(maxit,1);
kaspa=zeros(maxit,1); 
lbdm2xa=zeros(maxit,maxt);
lbdm2ya=zeros(maxit,maxt);
lbdm1a=zeros(maxit,maxt);
dhillona = zeros(maxit,1);
cspeca = zeros(maxit,1);

if strcmp(mode, 'kmeans')
    km_t = zeros(maxit,1);
end

lsct = zeros(maxit,1);
kaspt = zeros(maxit,1);
lbdm2xt = zeros(maxit,maxt);
lbdm2yt = zeros(maxit,maxt);
lbdm1t = zeros(maxit,maxt);
dhillont = zeros(maxit,1);
cspect = zeros(maxit,1);

if strcmp(affinity, 'cosine')
    fea = bsxfun(@rdivide, fea, sqrt(sum(fea.^2, 2)));
end

if strcmp(mode, 'random')
        % compute sigma using 7-nearest-neighbors
        t0 = cputime;
        W = EuDist2(fea, fea, false);
        dump = zeros(n,7);        
        idx = dump;
        for k = 1:7
            [dump(:,k),idx(:,k)] = min(W,[],2);
            temp = (idx(:,k)-1)*n+(1:n)';
            W(temp) = 1e100; 
        end
        
        sigma = mean(mean(dump(:,7)));
        km_t = cputime - t0;
end

for i = 1:maxit
    %common representatives

    t0 = cputime;
    if strcmp(affinity, 'cosine')
        distance = 'cosine';
    else 
        distance = 'sqEuclidean';
    end
    
    if strcmp(mode, 'kmeans')
        [lb, reps, ~, VAR] = litekmeans(fea, r, 'Distance', distance,...
            'replicates', initRes, 'maxiter', initIter, 'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100, 'clustersample', 0.1);
        km_t(i) = cputime - t0;
        lbcount = hist(lb, 1:r);
        cluster_sigma = sqrt(VAR ./ lbcount);
        sigma = mean(cluster_sigma);
        
    elseif strcmp(mode, 'random')
        % random sampling
        sample = randsample(length(gnd), r);
        reps = fea(sample,:);
        
    end
    
    % setting up tops
    fprintf('Iteration %d:\n', i);

    opts.r = r;
    opts.s = s;
    opts.reps = reps;
    if strcmp(mode, 'kmeans')
        opts.lbcount = lbcount;
    end
    opts.sigma = sigma;
    
    % added random sampling
    opts.mode = mode;
    
    %LSC
    t0 = cputime;res = lsc(fea, nlabel, opts); lsct(i) = cputime - t0;
    lsca(i) = bestacc(res, gnd);
    fprintf('LSC: %f\n', lsca(i));
    
    %KASP
    %opts.sigma = sigma;
    %opts.pre_label = lb;
    %t0 = cputime;res = KASP(fea, nlabel, r, opts); kaspt(i) = cputime - t0;
    %    kaspa(i) = bestacc(res, gnd);
    %fprintf('KASP: %f\n', kaspa(i));

    % MY ALGORITHM
    %[l1,l2,l3,l4,l5,l6,l7,l8,original_t(i),reps_t(i),all_t(i)] = bask_t(fea, nlabel, affinity, maxt, opts);
    [lbdm2xi, lbdm2yi, lbdm1i, dhilloni,cspeci] = lbdm_suite(fea, nlabel, affinity, maxt, opts);
    lbdm2xia = lbdm2xi.a;
    lbdm2xit = lbdm2xi.t;
    lbdm2yia = lbdm2yi.a;
    lbdm2yit = lbdm2yi.t;
    lbdm1ia  = lbdm1i.a;
    lbdm1it  = lbdm1i.t;
    dhillonia= dhilloni.a;
    dhillonit= dhilloni.t;
    cspecia  = cspeci.a;
    cspecit  = cspeci.t;
    
    % lbdm 2x and 2y, j is time step
    for j = 1:maxt
        lbdm2xa(i,j) = bestacc(lbdm2xia(:,j),gnd);
        lbdm2xt(i,j) = lbdm2xit(j);
        lbdm2ya(i,j) = bestacc(lbdm2yia(:,j),gnd);
        lbdm2yt(i,j) = lbdm2yit(j);
        lbdm1a(i,j) = bestacc(lbdm1ia(:,j),gnd);
        lbdm1t(i,j) = lbdm1it(j);
        fprintf('lbdm 2x, step size = %d: %f\n', j*2, lbdm2xa(i,j));
        fprintf('lbdm 2y, step size = %d: %f\n',j*2, lbdm2ya(i,j));
        fprintf('lbdm 1,  step size = %d: %f\n', j*2-1, lbdm1a(i,j));   
    end 
    
    % dhillon and cspec
    dhillona(i) = bestacc(dhillonia, gnd);
    dhillont(i) = dhillonit;
    cspeca(i) = bestacc(cspecia, gnd);
    cspect(i) = cspecit;
    
    fprintf('dhillon co-clusteri: %f\n', dhillona(i));
    fprintf('cspec: %f\n', cspeca(i));
      
    clear opts;
 
end
% Average accuracy
fprintf('\nAverage accuracy:\n')
fprintf('LSC: %f\n', mean(lsca));
%fprintf('KASP: %f\n', mean(kaspa));

for j = 1:maxt
    fprintf('lbdm2x, step size = %d: %f\n', j*2, mean(lbdm2xa(:,j)));
    fprintf('lbdm2y, step size = %d: %f\n',j*2, mean(lbdm2ya(:,j)));
    fprintf('lbdm1, step size = %d: %f\n', j*2-1, mean(lbdm1a(:,j)));
end
fprintf('dhillon co-clustering: %f\n', mean(dhillona));
fprintf('cspec: %f\n',mean(cspeca));

save(strcat(mat, '_bask_result'), 'lsca','lsct','lbdm2xa','lbdm2xt','lbdm2ya','lbdm2yt','lbdm1a','lbdm1t',...
    'dhillona','dhillont','cspeca','cspect','km_t')
clear;
end

function acc = bestacc(label, gnd)
l = bestMap(gnd, label);
acc = sum((l - gnd) == 0) / size(gnd, 1);
end
