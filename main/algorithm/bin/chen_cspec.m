function inds_sc = chen_cspec(A, k, p)
[n,m] = size(A);
indSmp = randperm(n);
landmarks = A(indSmp(1:p),:); % initial centroids

[kmeans_labels,landmarks,~,sumD,dists] = litekmeans(A, p,'MaxIter',5,'Replicates',1,'Start',landmarks);
dists = EuDist2(A, landmarks, false);

sigma2 = mean(mean(dists));
lambda = 1/sigma2;

Gsdx = exp(-dists*lambda);

D1 = sqrt(sum(Gsdx,1)); D1(D1<1e-12)=1e-12;
Gsdx = bsxfun(@rdivide, Gsdx, D1);
D2 = sqrt(sum(Gsdx,2)); D2(D2<1e-12)=1e-12;
Gsdx = bsxfun(@rdivide, Gsdx, D2);

[U,S] = svds(Gsdx, k);
temp = sqrt(sum(U.^2,2)); temp(temp<1e-12)=1e-12;
U = bsxfun(@rdivide, U, temp);
inds_sc = kmeans(U, k, 'replicates', 10);

end