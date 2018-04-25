addpath(genpath('/home/drproduck/Documents/MATLAB/'))
load('data_20news');
n = size(X, 1);
m = size(X, 2);

idx = [find(labels == 1); find(labels == 2); find(labels == 3); find(labels == 4)];
fea = X(idx, :);
gnd = labels(idx, :);
save('news4', 'fea','gnd','cls_names','words');