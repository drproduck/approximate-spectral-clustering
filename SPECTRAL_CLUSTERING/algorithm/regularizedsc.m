function label = regularizedsc(fea,k,r,s,t)
fileid=1;
[n,m] = size(fea);
fprintf(fileid,'picking landmarks...\n')
[lb, reps, ~, VAR] = litekmeans(fea, r,'MaxIter', 100, 'Replicates', 1,...
    'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100,...
    'clustersample', 0.1);
    
%determine sigma
lbcount = hist(lb, 1:r);
cluster_sigma = sqrt(VAR ./ lbcount);
sigma = mean(cluster_sigma); 
fprintf(fileid,'using sigma = %.2f\n',sigma);
 
%sparse representation
fprintf(fileid,'constructing sparse A...\n');
W = EuDist2(fea, reps, 0);
tic;
if s > 0
    dump = zeros(n,s);
    idx = dump;
    for i = 1:s
        [dump(:,i),idx(:,i)] = min(W,[],2);
        temp = (idx(:,i)-1)*n+(1:n)';
        W(temp) = 1e100; 
    end
    
    % manipulate index to efficiently create sparse matrix Z
    Gidx = repmat((1:n)',1,s);
    Gjdx = idx;
    W = sparse(Gidx(:),Gjdx(:),dump(:),n,r);

elseif s <= 0 % default to dense matrix
    fprintf(fileid,'default to dense matrix\n');
    [~,idx] = min(W,[],2);
    W = exp(-W/(2.0*sigma^2));
end

fprintf(fileid,'Computing Laplacian and diffusion map...\n');
d1 = sum(W, 2);
d1=d1+sum(d1,1)/n;
d2 = sum(W, 1);
d2=d2+sum(d2,2)/r;
D1 = sparse(1:n,1:n,d1.^(-0.5));
D2 = sparse(1:r,1:r,d2.^(-0.5));
L = D1*W*D2;
[U,S,V] = svds(L, k);

if t > 0
    U = D1 * U * S.^t;
    V = D2 * V * S.^t;
elseif t == 0
    U = D1 * U;
    V = D2 * V;
end

U(:,1)=[];
V(:,1)=[];
U=U./sqrt(sum(U.^2,2));
V=V./sqrt(sum(V.^2,2));

% label=litekmeans(U, k, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates',10);
reps_labels=litekmeans(V, k, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates',10);
label = zeros(n, 1);
for i = 1:n
    can_reps_label = reps_labels(idx(i,:));
    [~, m] = max(hist(can_reps_label, 1:k));
    label(i) = m;
end

end
