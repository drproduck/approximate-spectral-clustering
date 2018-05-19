load('20newsorigin.mat')
a=(fea~=0);
col_sum=sum(a,1);
[val,idx] = sort(col_sum,'descend');
figure(1)
scatter(1:1000, val(1:1000), 1,'Marker','.');
text(50:100, val(50:100), vocab(idx(50:100)), 'FontSize',7);

col_remove=[find(col_sum<=1),find(col_sum>=3000)];
fea(:,col_remove)=[];
vocab(col_remove)=[];

row_sum=sum(fea,2);
row_remove=find(row_sum==0);
fea(row_remove,:)=[];
gnd(row_remove)=[];
% fea=double(fea~=0);

save('20newstruncated','fea','gnd','vocab','topic')
