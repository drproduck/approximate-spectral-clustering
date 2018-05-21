%ORIGIN: all words are used
% visualizing word and doc embedding in bipartite Laplacian, 20news 
clear;
colormap default
load('20newstruncated.mat')
nlabel=max(gnd);
[n,m] = size(fea);
% if using all groups
% fea = min(fea, 10);
fea = tfidf(fea,'hard');
fea=fea./sqrt(sum(fea.^2,2));
% idx = [find(gnd==1);find(gnd==2);find(gnd==3);find(gnd==4);find(gnd==5)];
idx=[find(gnd==1);find(gnd==2);find(gnd==10);find(gnd==13);find(gnd==14)];
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
% figure(1);
% row_sum = sum(fea,2);
% col_sum=sum(fea,1);
% [val,~]=sort(row_sum);
% subplot(121)
% scatter(1:n,val)
% [val,~]=sort(col_sum);
% subplot(122)
% scatter(1:m,val)
% drawnow
%

% figure(2)
% % no regularizer
% [L,D1,D2] = getLaplacian(fea, 1e-100, 'bipartite','r',[0,0]);
% [U,S,V] = svds(L, 20);
% for t=0:0
%     if t==0
%         U = D1 * U;
%         V = D2 * V;
%     elseif t~=0
%         U=D1*U*S.^t;
%         V=D2*V*V.^t;
%     end
%     U(:,1)  = [];
%     V(:,1) = [];
%     U=U./sqrt(sum(U.^2,2));
%     V=V./sqrt(sum(V.^2,2));
%     disp('no reuglarize accuracy:')
% %     ac=embedcluster(U,V,20,gnd,'kmeans')
% end
% % subplot(121)
% % scatter3(V(:,1), V(:,2), V(:,3), 1);
% scatter3(V(idx_rw,1),V(idx_rw,2),V(idx_rw,3),1,'Marker','.');
% text(V(idx_rw,1), V(idx_rw,2), V(idx_rw,3), vocab(idx_rw), 'FontSize',7);
% hold on
% U = U(idx,:);
% % sample some docs only
% % idx = randsample(length(gnd), 500);
% % scatter3(U(idx,1), U(idx,2), U(idx,3), 3, gnd(idx));
% 
% scatter3(U(:,1), U(:,2), U(:,3), 3, gnd1, 'Marker','o');


% average regularizer
% subplot(122)
[L,D1,D2] = getLaplacian(fea, 1e-100, 'bipartite');
[u,s,v] = svds(L, nlabel);
for t=0
    if t==0
        U = D1 * u;
        V = D2 * v;
    elseif t>=0
        U=D1*u*s.^t;
        V=D2*v*s.^t;
    end
    U(:,1) = [];
    V(:,1) = [];
    U=U./sqrt(sum(U.^2,2));
    V=V./sqrt(sum(V.^2,2));
    disp('regularize accuracy:')
    [label,ac]=embedcluster(U,V,gnd,'kmeans','coclustering');
    ac
end
%
% get most common words from each ground truth groups
idx_rw_1 = getMostCommon(fea, gnd, 1, 5);
idx_rw_2 = getMostCommon(fea, gnd, 2, 5);
idx_rw_3 =getMostCommon(fea,gnd,3,5);
idx_rw_4 =getMostCommon(fea,gnd,4,5);
idx_rw_5 =getMostCommon(fea,gnd,5,5);
idx_rw_10=getMostCommon(fea,gnd,10,5);
idx_rw_13=getMostCommon(fea,gnd,13,5);
idx_rw_14=getMostCommon(fea,gnd,14,5);


% idx_rw = [idx_rw_1;idx_rw_2;idx_rw_3;idx_rw_4;idx_rw_5];
idx_rw=[idx_rw_1;idx_rw_2;idx_rw_10;idx_rw_13;idx_rw_14];
vocab(idx_rw_1)
vocab(idx_rw_2)
vocab(idx_rw_3)
vocab(idx_rw_4)
vocab(idx_rw_5)
% idx_rw = randsample(length(vocab), 100);
%



% figure(2)
% [~,proj]=pca([U;V]);
% U=proj(1:n,:);
% V=proj(n+1:end,:);
% scatter3(V(idx_rw,1),V(idx_rw,2),V(idx_rw,3),1,'Marker','.');
% text(V(idx_rw,1), V(idx_rw,2),V(idx_rw,3), vocab(idx_rw), 'FontSize',5);
% hold on
% scatter3(U(idx,1),U(idx,2),U(idx,3), 3,bestColor(gnd1), 'Marker','o');
% drawnow

fig = gcf;
fig.PaperPositionMode = 'auto'
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

hold on
% [~,proj]=pca([U;V]);
% U=proj(1:n,:);
% V=proj(n+1:end,:);
g=[1,2,10,13,14];
color=distinguishable_colors(20);
for i=1:length(g)
    gn=find(gnd==g(i));
    scatter3(U(gn,1),U(gn,2),U(gn,3),3,color(i,:),'filled','Marker','o');
    hold on
end
text(V(idx_rw,1), V(idx_rw,2), V(idx_rw,3),vocab(idx_rw), 'FontSize',10);
legend(topic{g});
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
hold off

% scatter(V(idx_rw,1),V(idx_rw,2),1,'Marker','.');
% text(V(idx_rw,1), V(idx_rw,2),vocab(idx_rw), 'FontSize',5);
% hold on
% scatter(U(idx,1),U(idx,2),3,bestColor(gnd1), 'Marker','o');
% drawnow
% figure(3)
% scatter(V(idx_rw,1),V(idx_rw,2),1,'Marker','.');
% text(V(idx_rw,1), V(idx_rw,2),vocab(idx_rw), 'FontSize',10);
% hold on
% scatter(U(idx,1),U(idx,2), 3,bestColor(gnd1), 'Marker','o');
% drawnow
% hold off


% hold off
% figure(2)
% color=distinguishable_colors(20);
% for i=1:20
% scatter3(U(gnd==i,1),U(gnd==i,2),U(gnd==i,3),3,color(i,:),'Marker','o');
% hold on
% end
% legend(topic{:})

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

function [label,ac]=embedcluster(U,V,gnd,cluster_method,embed_method)
nlabel=max(gnd);
n=size(U,1);
W = [U;V];
if strcmp(cluster_method, 'kmeans')
    if strcmp(embed_method, 'coclustering')
        all_label = litekmeans(W, nlabel, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates',10);
        label = all_label(1:n);
    elseif strcmp(embed_method, 'direct')
        label=litekmeans(U,nlabel,'Distance','cosine','MaxIter',100,'Replicates',10);
    end
elseif strcmp(cluster_method, 'discretize')
    all_label = discretize(W);
    label = all_label(1:n);
end

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