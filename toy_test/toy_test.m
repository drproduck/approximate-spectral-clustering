clear;
addpath('deng cai/');
addpath('dataset/');
addpath('dataset/paper_data');
addpath('pickled/');
load('circledata_50.mat');

if exist('gnd', 'var')
    nlabel = max(gnd)
else
    nlabel = 3;
end

sigma = 20;

figure(11)
scatter(fea(:,1), fea(:,2), [], gnd);

%NJW
% opts.sigma = sigma;
% [labels, ~] = SC(fea, nlabel, 'gaussian', opts);
% if exist('gnd', 'var')
%     labels = bestMap(gnd, labels);
%     ac = sum(labels == gnd) / size(gnd, 1);
%     fprintf('sc ac: %d\n', ac);
% end
% % [~, arg] = sort(labels);
% % spy(W(arg,arg))
% figure(1)
% scatter(fea(:,1),fea(:,2), [], labels)
% clear opts
% gnd = labels;

% sparse symmetric SC
% opts.sigma = sigma;
% opts.sparse = 0.05;
% opts.mode = 'local';
% [labels, L] = SC(fea, nlabel, 'gaussian', opts);
% if exist('gnd', 'var')
%     labels = bestMap(gnd, labels);
%     ac = sum(labels == gnd) / size(gnd, 1);
%     fprintf('sc ac: %d\n', ac);
% end
% [~, arg] = sort(labels);
% spy(L(arg,arg))
% figure(2)
% scatter(fea(:,1),fea(:,2), [], labels)
% clear opts

% sparse assymmetric SC
% opts.sigma = sigma;
% opts.sparse = 100;
% [labels, L] = ASC(fea, nlabel, 'gaussian', opts);
% if exist('gnd', 'var')
%     labels = bestMap(gnd, labels);
%     ac = sum(labels == gnd) / size(gnd, 1);
%     fprintf('sparse sc ac: %d\n', ac);
% end
% [~, arg] = sort(labels);
% spy(L(arg,arg))
% figure(3)
% scatter(fea(:,1),fea(:,2), [], labels)
% clear opts;
% gnd = labels;
% 
% % LSC 
r = 100;
% 
[lb, reps, ~, VAR] = litekmeans(fea, r, 'MaxIter', 10, 'Replicates', 1);
lbcount = hist(lb, 1:r);
sigma = mean(sqrt(VAR ./ lbcount));
figure(10)
scatter(reps(:,1), reps(:,2),[],'x', 'red')

% % 
% opts.r = 10;
% opts.p = r;
% opts.sigma = sigma;
% opts.reps = reps;
% [labels, X] = LSC(fea, nlabel, opts);
% if exist('gnd', 'var')
%     labels = bestMap(gnd, labels);
%     ac = sum(labels == gnd) / size(gnd, 1);
%     fprintf('lsc ac: %d\n', ac);
% end
% figure(4)
% scatter(fea(:,1),fea(:,2), [], labels)
% clear opts;
% 
% % BIAS 
opts.r = r;
opts.sigma = sigma;
opts.sparse = 10;
opts.kasp = 1;
opts.reps = reps;
% opts.t = 2;
opts.lbcount = lbcount;
[labels, nearest_labels, reps_labels,U,V] = BIASy(fea, nlabel, 'gaussian', opts);
nearest_labels;
if exist('gnd', 'var')
    labels = bestMap(gnd, labels);
    ac = sum(labels == gnd) / size(gnd, 1);
    fprintf('bias ac: %d\n', ac);
end
figure(5)
scatter(U, zeros(size(U,1), 1), [], labels)

figure(6)
scatter(fea(:,1), fea(:,2), [], labels);


clear opts;

%KASP
% opts.reps = reps;
% opts.sigma = 1;
% opts.pre_label = lb;
% [labels] = KASP(fea, nlabel, r, opts);
% if exist('gnd', 'var')
%     labels = bestMap(gnd, labels);
%     ac = sum(labels == gnd) / size(gnd, 1);
%     fprintf('kasp ac: %d\n', ac);
% end
% figure(7)
% scatter(fea(:,1),fea(:,2), [], labels)

% BIASx 
% opts.r = r;
% opts.sigma = sigma;
% opts.sparse = 3;
% opts.kasp = 1;
% opts.reps = reps;
% opts.t = 5;
% opts.lbcount = lbcount;
% [labels, Y] = BiASx(fea, nlabel, 'gaussian', opts);
% if exist('gnd', 'var')
%     labels = bestMap(gnd, labels);
%     ac = sum(labels == gnd) / size(gnd, 1);
%     fprintf('bias ac: %d\n', ac);
% end
% clear opts;

%kmeans
% lb = litekmeans(fea, nlabel, 'MaxIter', 100, 'Replicates', 10);
% ac = sum(lb == gnd) / size(gnd, 1)
% figure(8)
% scatter(fea(:,1), fea(:,2), [], lb);







