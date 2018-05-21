%ORIGIN: all words are used
% visualizing word and doc embedding in bipartite Laplacian, 20news 
clear;
colormap default
load('20newsorigintop10.mat')
nlabel=max(gnd);

[n,m] = size(fea);
% fea = min(fea, 10);
fea=tfidf(fea,'hard');
fea=fea./sqrt(sum(fea.^2,2));

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
scatter(1:length(gnd),val)
[val,~]=sort(col_sum);
subplot(122)
scatter(1:m,val)
% drawnow
%

%
% get most common words from each ground truth groups
[idx_rw_1,idx_doc_1] = getMostCommon(fea, gnd, 1, 20);
[idx_rw_2,idx_doc_2] = getMostCommon(fea, gnd, 2, 20);
[idx_rw_3,idx_doc_3] =getMostCommon(fea,gnd,3,20);
[idx_rw_4,idx_doc_4] = getMostCommon(fea,gnd,4,20);
[idx_rw_5,idx_doc_5] = getMostCommon(fea,gnd,5,20);
[idx_rw_6,idx_doc_6] = getMostCommon(fea,gnd,6,20);
idx_rw = [idx_rw_1;idx_rw_2;idx_rw_3];
idx_doc=[idx_doc_1;idx_doc_2;idx_doc_3];
vocab(idx_rw_1)
vocab(idx_rw_2)
vocab(idx_rw_3)


% average regularizer
% subplot(122)
[L,D1,D2] = getLaplacian(fea, 1e-100, 'bipartite');
[u,s,v] = svds(L,nlabel);
for t=0:0
    if t==0
        U = D1 * u;
        V = D2 * v;
    elseif t>=0
        U=D1*u*s.^t;
        V=D2*v*s.^t;
    end
    U(:,1)  = [];
    V(:,1) = [];
    U=U./sqrt(sum(U.^2,2));
    V=V./sqrt(sum(V.^2,2));
    disp('regularize accuracy:')
    ac=embedcluster(U,V,nlabel,gnd,'kmeans')
end

% [~,proj]=pca([U;V]);
% U=proj(1:n,:);
% V=proj(n+1:end,:);
figure(2)
scatter3(V(idx_rw,1),V(idx_rw,2),V(idx_rw,3),1,'Marker','.');
text(V(idx_rw,1), V(idx_rw,2), V(idx_rw,3),vocab(idx_rw), 'FontSize',7);
hold on
scatter3(U(idx_doc,1),U(idx_doc,2), U(idx_doc,3),3,bestColor(gnd(idx_doc)), 'Marker','o');
drawnow


% fast cosine + outlier removal
% ac = zeros(10,1);
% ix = 0.01:0.01:0.1;
% i = 1;
% for remove_low = ix
%     fprintf('remove %.2f%% lowest points\n',remove_low*100)
%     ac(i) = cosine(fea, nlabel,gnd,remove_low);
%     i = i + 1;
% end
% plot(ix, ac);

% kmeans with landmark and outlier removal
% ac = zeros(10,1);
% ix = 0.01:0.01:0.1;
% i = 1;
% for remove_low = ix
%     fprintf('remove %.2f%% lowest points\n',remove_low*100)
%     ac(i) = landmark(fea, nlabel,gnd,remove_low);
%     i = i + 1;
% end
% plot(ix, ac);
%

%
% this code to plot points along dimensions
% for d=1:2:18
% fig=figure(3);
% scatter(V(idx_rw,d),V(idx_rw,d+1),1,'Marker','.');
% text(V(idx_rw,d), V(idx_rw,d+1), vocab(idx_rw), 'FontSize',5);
% hold on
% scatter(U(:,d),U(:,d+1), 3,bestColor(gnd1), 'Marker','o');
% saveas(fig,strcat('ng2_ng3_ng4_',num2str(d),'_',num2str(d+1),'.jpg'));
% drawnow
% hold off
% end


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

function [idx_rw,idx_doc] = getMostCommon(fea, gnd, z, no_keep)
% get most common words based on column sums, tfidf weight, assuming ground
% truth (gnd) is known
idx_doc=find(gnd==z);
col_sum = sum(fea(idx_doc,:), 1);
[~, idx_max] = sort(col_sum, 'descend');
idx_rw = idx_max(1:no_keep);
end

%% kmeans landmark + outlier removal
function ac = landmark(fea,nlabel, gnd,remove_low)
[label, kept_idx] = BASK(fea, nlabel, 500,5,2, 'cosine', 'remove_outlier',...
    [0,remove_low],'select_method','kmeans', 'embed_method', 'direct');
gnd = gnd(kept_idx);
label = bestMap(gnd, label);
ac = sum(label == gnd) / size(gnd, 1)
end

%% doc_term + outlier removal
function ac = doc_term(fea,nlabel,gnd,remove_low)
[label,kept_idx] = fast_cosine_SC(fea, nlabel, 2,'cluster_method','kmeans','remove_low',remove_low,'embed_method','coclustering',...
    'remove_high',1,'affinity_type','doc_term');
gnd = gnd(kept_idx);
label = bestMap(gnd, label);
ac = sum(label == gnd) / size(gnd, 1)
end

%% fast cosine + outlier removal
function ac = cosine(fea, nlabel, gnd,remove_low)
[label,kept_idx] = fast_cosine_SC(fea, nlabel, 0,'cluster_method','kmeans','remove_low',remove_low,...
    'embed_method','direct','remove_high',1,'affinity_type','point');
gnd = gnd(kept_idx);
size(label)
label = bestMap(gnd, label);
ac = sum(label == gnd) / size(gnd, 1)
end