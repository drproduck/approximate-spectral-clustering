% load('embeddings.mat')
% size(U)
% size(V)
% w=[U;V];
% p=w*w';
% d=sum(p,2);
% p=p./d;
% x=tsne_p(p,[],2);

load('20newsorigin')
[n,m]=size(fea);
fea = tfidf(fea,'hard');
fea=fea./sqrt(sum(fea.^2,2));
load('20news_tsne')
idx=[find(gnd==1);find(gnd==2);find(gnd==3);find(gnd==4);find(gnd==5)];
gnd1=gnd(idx);

idx_rw_1 = getMostCommon(fea, gnd, 1, 5);
idx_rw_2 = getMostCommon(fea, gnd, 2, 5);
idx_rw_3 =getMostCommon(fea,gnd,3,5);
idx_rw_4 =getMostCommon(fea,gnd,4,5);
idx_rw_5 =getMostCommon(fea,gnd,5,5);
idx_rw_10=getMostCommon(fea,gnd,10,5);
idx_rw_13=getMostCommon(fea,gnd,13,5);
idx_rw_14=getMostCommon(fea,gnd,14,5);


idx_rw = [idx_rw_1;idx_rw_2;idx_rw_3;idx_rw_4;idx_rw_5];
% idx_rw=[idx_rw_1;idx_rw_2;idx_rw_13;idx_rw_14];
vocab(idx_rw_1)
vocab(idx_rw_2)
vocab(idx_rw_3)
vocab(idx_rw_4)
vocab(idx_rw_5)
U=y(1:n,:);
V=y(n+1:end,:);
clear y;
text(V(idx_rw,1), V(idx_rw,2),vocab(idx_rw), 'FontSize',10);
hold on
scatter(U(idx,1),U(idx,2),3,bestColor(gnd1),'filled','Marker','o')


function idx_rw = getMostCommon(fea, gnd, z, no_keep)
% get most common words based on column sums, tfidf weight, assuming ground
% truth (gnd) is known
col_sum = sum(fea(gnd==z,:), 1);
[~, idx_max] = sort(col_sum, 'descend');
idx_rw = idx_max(1:no_keep);
end
