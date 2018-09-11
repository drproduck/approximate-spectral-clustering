function sigma = estimate_sigma_knn(X, k, nSample)
% estimate parameter sigma for features X, Euclidean distance

% check if number of datapoints less than number of samples 
n = size(X,1);
if n <= nSample
   nSample = n
end

X = X(randsample(n, nSample), :);
W = EuDist2(X, X, 0);

dump = zeros(nSample, k);
for j = 1:k
    [dump(:,j),idx] = min(W,[],2);
    temp = (idx-1)*nSample+(1:nSample)';
    W(temp) = 1e+100; 
end

sigma = mean(mean(dump));