clear;
mat = 'mnist';
load(mat);
nlabel = max(gnd);
rng(99999);
maxit = 20;
id = fopen('dump');
acc = zeros(maxit,1);

for i = 1:maxit
    [label,idx] = BASK(fea, nlabel, 500, 5, 0, 'cosine','remove_outlier',[0.3 1],'fileid',id);
    label = bestMap(gnd, label);
    ac = sum(label == gnd) / size(gnd, 1);
    acc(i) = ac;
    fprintf('run %d accuracy = %.2f%%\n',i,ac*100);
end
% figure(2)
% c = zeros(2000,1);
% c(idx) = 1;
% scatter(fea(:,1),fea(:,2),[],c)
fprintf('average accuracy = %.2f%%\n',mean(acc)*100);
save(strcat('accuracy_',mat),'acc');
fclose(id);