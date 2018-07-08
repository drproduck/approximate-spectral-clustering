addpath(genpath('/home/drproduck/Documents/MATLAB/SPECTRAL_CLUSTERING'));
% mat = {'letter' 'mnist' 'usps' 'protein' 'pend' 'shuttle'};
k = 7;
mat = {'shuttle'};

for i = 1:1
    load(mat{i});
    fprintf('getting samples\n')
    fea = subsample(fea, 0.1);
    fprintf('sample size = %d\n', size(fea, 1))
    fprintf('Finding sigma on %s\n', mat{i}); 
    t0 = cputime;
    sigma = sigma_from_knn(fea, k);
    t = cputime - t0;
    fprintf('sigma = %f\n', sigma)
    fprintf('time = %f\n', t)
end

function sigma = sigma_from_knn(fea, k)

%Calculate sigma by taking the average distance of all points to their
%kth-nearest neighbor

W = EuDist2(fea, fea, true);
n = size(fea, 1);
s = k + 1; %ignoring the data point itself
minStore = zeros(n, s);
idxStore = zeros(n, s);
for i = 1:s
    [minStore(:,i), idxStore(:,i)] = min(W, [], 2);
    temp = (idxStore(:,i)-1)*n+(1:n)';
    W(temp) = 1e100;
end

sigma = mean(minStore(:,s), 1);
end

function subfea = subsample(fea, ratio)
n = size(fea, 1);
m = floor(ratio * n);
subfea = fea(randsample(n, m),:);

end

