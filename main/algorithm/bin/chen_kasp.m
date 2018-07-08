function inds_sc = chen_kasp(A , k, p)
[n,m] = size(A);
indSmp = randperm(n);
landmarks = A(indSmp(1:p),:); % initial centroids

[kmeans_labels,landmarks,~,sumD,dists] = litekmeans(A, p,'MaxIter',5,'Replicates',1,'Start',landmarks);

[inds_lmk, labels_out] = specluster(landmarks, k, struct('clustering', 'NJW', 'affinity', 'Gaussian'));
inds_lmk(inds_lmk==0) = labels_out;
inds_sc = inds_lmk(kmeans_labels);
end