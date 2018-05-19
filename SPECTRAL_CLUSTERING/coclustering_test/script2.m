%10879: remove least common words, keep only 10879 most common words, stop
%words also removed
% visualizing word and doc embedding in bipartite Laplacian, 20news
clear;
colormap default
load('20news10879')

[n,m] = size(fea);
% if using all groups
% fea = min(fea, 10);
fea = tfidf(fea);
fea=fea./sqrt(sum(fea.^2,2));
idx = [find(gnd == 1); find(gnd == 2)];
gnd1 = gnd(idx,:);

% if using only indicated groups
% fea = min(fea, 10);
% idx = [find(gnd==1);find(gnd==2)];
% fea=fea(idx,:);
% fea=tfidf(fea);
% gnd=gnd(idx,:);

if sum(fea ~= 0, 1) == 0
    disp('null column exists')
end
if sum(fea ~= 0, 2) == 0
    disp('null row exists')
end

%
% draw column sum and row sum distribution
figure(1);
row_sum = sum(fea,2);
col_sum=sum(fea,1);
[val,~]=sort(row_sum);
subplot(121)
scatter(1:n,val)
[val,~]=sort(col_sum);
subplot(122)
scatter(1:m,val)
%

%
% get most common words from each ground truth groups
idx_rw_1 = getMostCommon(fea, gnd, 1, 20);
idx_rw_2 = getMostCommon(fea, gnd, 2, 20);
idx_rw_3 =getMostCommon(fea,gnd,3,20);
idx_rw_4 = getMostCommon(fea,gnd,4,20);

idx_rw = [idx_rw_1;idx_rw_2];
% idx_rw = randsample(length(vocab), 100);
%

figure(2)
% no regularizer
[L,D1,D2] = getLaplacian(fea, 1e-100, 'bipartite','r',[0,0]);
[U,S,V] = svds(L, 20);
U = D1 * U;
V = D2 * V;
% 
U(:,1) = [];
V(:,1) = [];
U=U./sqrt(sum(U.^2,2));
V=V./sqrt(sum(V.^2,2));
disp('no reuglarize accuracy:')
ac=embedcluster(U,V,20,gnd,'kmeans')
subplot(121)
% scatter3(V(:,1), V(:,2), V(:,3), 1);
scatter3(V(idx_rw,1),V(idx_rw,2),V(idx_rw,3),1,'Marker','.');
text(V(idx_rw,1), V(idx_rw,2), V(idx_rw,3), vocab(idx_rw), 'FontSize',7);
hold on
U = U(idx,:);
% sample some docs only
% idx = randsample(length(gnd), 500);
% scatter3(U(idx,1), U(idx,2), U(idx,3), 3, gnd(idx));

scatter3(U(:,1), U(:,2), U(:,3), 3, gnd1, 'Marker','o');

% average regularizer
subplot(122)
[L,D1,D2] = getLaplacian(fea, 1e-100, 'bipartite');
[U,S,V] = svds(L, 20);
U=D1*U;
V=D2*V;
U(:,1)=[];
V(:,1)=[];
U=U./sqrt(sum(U.^2,2));
V=V./sqrt(sum(V.^2,2));
disp('regularize accuracy:')
ac=embedcluster(U,V,20,gnd,'kmeans')

scatter3(V(idx_rw,1),V(idx_rw,2),V(idx_rw,3),1,'Marker','.');
text(V(idx_rw,1), V(idx_rw,2), V(idx_rw,3), vocab(idx_rw), 'FontSize',7);
hold on
U = U(idx,:);
scatter3(U(:,1), U(:,2), U(:,3), 3, gnd1, 'Marker','o');

%
% dead word
% col_sum = sum(fea ~= 0, 1);
% idx = find(col_sum == 0);
% length(idx)
% scatter(V(idx,1), V(idx,2), 1000, 'red','Marker','x')
% idx = idx(1:3);
% text(V(idx,1), V(idx,2), vocab(idx));
% vocab(idx)
%

%
% lone word
% cs1=sum(fea(gnd==1,:)~=0,1);
% cs2=sum(fea(gnd==2,:)~=0,1);
% idx1 = intersect(find(cs2==0),find(cs1>0));
% idx2 = intersect(find(cs1==0),find(cs2>0));
% idx3 = intersect(find(cs1>0),find(cs2>0));
% length(idx1)
% length(idx2)
% length(idx3)
% scatter(V(idx1,1), V(idx1,2), 10, 'red','Marker','x')
% scatter(V(idx2,1), V(idx2,2), 10, 'green','Marker','d')
% scatter(V(idx3,1), V(idx3,2), 10, 'yellow','Marker','o')
%

% g1 = sum(fea(gnd == 1,:) ~= 0, 1);
% g2 = sum(fea(gnd == 2,:) ~= 0, 1);
% close_idx = find(abs(g1 - g2) == 0);
% length(close_idx)
% text(V(close_idx,1), V(close_idx,2), vocab(close_idx)) 

%
% to label some words that seem out of place
% 
% w_idx = find(V(:,2) >= 0.008);
% text(V(w_idx,1), V(w_idx,2), vocab(w_idx))
% 
%

%
% this code was used to find centroids and their closest words
%
% center1 = mean(U(gnd == 1,:), 1);
% center2 = mean(U(gnd == 2,:), 1);
% center = [center1;center2];
% scatter(center(:,1), center(:,2), 36, 'red', 'Marker', 'x')
% 
% center1_to_word = EuDist2(center1, V, 0);
% center2_to_word = EuDist2(center2, V, 0);
% 
% no = 10;
% [~,idx1] = sort(center1_to_word, 'ascend');
% text(V(idx1(1:no),1), V(idx1(1:no),2), vocab(idx1(1:no)));
% vocab(idx1(1:no))
% [~,idx2] = sort(center2_to_word, 'ascend');
% text(V(idx2(1:no),1), V(idx2(1:no),2), vocab(idx2(1:no)));
% vocab(idx2(1:no))
%

%%%
%
% this code was used to vary diffusion time step t
%
% for t = 0:1:10
%     if t == 0
%         U = D1 * u;
%         V = D2 * v;
%     else
%         U = D1 * u * s.^t;
%         V = D2 * v * s.^t;
%     end
% 
%     U(:,1) = [];
%     V(:,1) = [];
%     scatter(U(:,1), U(:,2), 1, gnd);
%     drawnow
%     pause(2);
% end
%
%%%

function ac=embedcluster(U,V,k,gnd,cluster_method)
n=size(U,1);
W = [U;V];
if strcmp(cluster_method, 'kmeans')
    all_label = litekmeans(W, k, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates',10);
elseif strcmp(cluster_method, 'discretize')
    all_label = discretize(W);
end
label = all_label(1:n);
label=bestMap(gnd,label);
ac=sum(label==gnd)/length(gnd);
end

function idx_rw = getMostCommon(fea, gnd, z, no_keep)
% get most common words based on column sums, tfidf weight, assuming ground
% truth (gnd) is known
col_sum = sum(fea(gnd==z,:), 1);
[~, idx_max] = sort(col_sum, 'descend');
idx_rw = idx_max(1:no_keep);
end