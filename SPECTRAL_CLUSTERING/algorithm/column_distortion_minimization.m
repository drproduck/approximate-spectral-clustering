% Cluster columns of the full affinity matrix. Reduced matrix derived from
% embedding lemma

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