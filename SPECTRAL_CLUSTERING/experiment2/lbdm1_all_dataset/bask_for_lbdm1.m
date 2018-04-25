function [lbdm1_acc, lbdm1_t] = bask_for_lbdm1(fea, k, affinity, t, opts)
%This function is ONLY for lbdm1. This was part of the address to
%reviewer's comment. Textgraph 2018

t0 = cputime;
n = size(fea, 1);

if (~exist('opts','var'))
   opts = [];
end

% number of representative points

if isfield(opts, 'r')
    r = opts.r;
    fprintf('# landmarks = %d\n', r)
    if r ~= 500
        warning('r should be 500 for this test')
    end
else
    error('r must be specified (testing function)')
end    
    
if strcmp(affinity, 'gaussian')
    reps = opts.reps;
    W = EuDist2(fea, reps, 0);
    
    if isfield(opts, 'sigma')
        sigma = opts.sigma;
        fprintf('sigma = %f\n', sigma)
    else
        error('Sigma must be specified (testing function)')
    end
 
    % sparse representation
    if isfield(opts, 's')
        s = opts.s;
        fprintf('# nearest landmarks (sparsity) = %d\n', s)
        
        dump = zeros(n,s);
        idx = dump;
        for i = 1:s
            [dump(:,i),idx(:,i)] = min(W,[],2);
            temp = (idx(:,i)-1)*n+(1:n)';
            W(temp) = 1e100; 
        end

        % manipulate index to efficiently create sparse matrix Z
        % Z is now (normalized to sum 1) smallest r landmarks in each row
        dump = exp(-dump/(2.0*sigma^2));
        Gidx = repmat((1:n)',1,s);
        Gjdx = idx;
        W = sparse(Gidx(:),Gjdx(:),dump(:),n,r);
        
    end
    
else
    error('unsupported affinity function')
end

d1 = sum(W, 2);
d1 = max(d1, 1e-100);
d2 = sum(W, 1);
d2 = max(d2, 1e-100);

D1 = sparse(1:n, 1:n, d1.^(-0.5));
D2 = sparse(1:r, 1:r, d2.^(-0.5));

L = D1 * W * D2;

[U,S,V] = mySVD(L, k);

finalIter = 100;
finalRes = 10;
if isfield(opts, 'finalIter')
    finalIter = opts.finalIter;
end
if isfield(opts, 'finalRes')
    finalRes = opts.finalRes;
end

% save a copy of U and V
UN = U;
VN = V;
% different ways to do clustering

base_t = cputime - t0;
lbdm1_acc = zeros(n, t);
lbdm1_t = zeros(t, 1);

for T = 1:t
    % 1
    fprintf('alpha = %d\n', T)
    t0 = cputime;
    U = D1 * UN * S.^(2*T-1);
    V = D2 * VN * S.^(2*T-1);
    W = [U;V];
    W(:,1) = [];
    W = bsxfun(@rdivide, W, sqrt(sum(W.^2, 2))); 
    all_label = litekmeans(W, k, 'Distance', 'cosine', 'MaxIter', finalIter, 'Replicates',finalRes);
    lbdm1_acc(:,T) = all_label(1:n);
    lbdm1_t(T) = cputime - t0 + base_t;
end

end