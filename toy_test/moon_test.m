clear;
addpath('deng cai/');
addpath('dataset/');
addpath('pickled/');
load('circledata_50.mat');

opts.sigma = 20;
[labels, U] = SC(fea, 2, 'gaussian', opts);
labels = bestMap(gnd, labels);
ac = sum(labels == gnd) / size(gnd, 1);
fprintf('sc ac: %d\n', ac);
figure(1)
scatter(fea(:,1),fea(:,2), [], labels)
clear opts

opts.r = 3;
opts.p = 10;
opts.sigma = 20;
[labels, X] = LSC(fea, 2, opts);
labels = bestMap(gnd, labels);
ac = sum(labels == gnd) / size(gnd, 1);
fprintf('lsc ac: %d\n', ac);
figure(2)
scatter(fea(:,1),fea(:,2), [], labels)
clear opts;

opts.r = 10;
% opts.sparse = 3;
opts.kasp = 1;
opts.sigma = 20;
[labels, Y] = BiAS(fea, 2, 'gaussian', opts);
figure(3)
% [U, X, Y]
labels = bestMap(gnd, labels);
ac = sum(labels == gnd) / size(gnd, 1);
fprintf('bias ac: %d\n', ac);
% scatter(Y(:,1), Y(:,2), [], gnd)
figure(3)
scatter(fea(:,1),fea(:,2), [], labels)

W = EuDist2(fea, fea, 0);
sigma = 20;
W = exp(-W/(2.0*sigma^2));

[labels, reps] = litekmeans(W', 10, 'MaxIter', 100, 'Replicates', 5);
for i = 1 : size(fea, 1)
    if i==1
        W = reps(labels(1),:);
    else 
        W = [W; reps(labels(i),:)];
    end
end

size(reps)
W = W';

%njw
n = size(W, 1);
D1 = sum(W, 2);
D2 = sum(W, 1);
D1 = max(D1, 1e-12);
D2 = max(D2, 1e-12);
D1 = sparse(1:n, 1:n, D1.^(-0.5));
D2 = sparse(1:n, 1:n, D2.^(-0.5));
L = D1 * W * D2;



[U, s, ~] = mySVD(L, 2);
% U(:,1) = [];
U = normr(U);

MaxIter = 10;
Replicates = 5;

res = litekmeans(U, 2, 'MaxIter', MaxIter, 'Replicates', Replicates);
res = bestMap(gnd,res);
ac = sum((res - gnd) == 0) / size(gnd, 1)
figure(4)
scatter(fea(:,1),fea(:,2), [], res)





