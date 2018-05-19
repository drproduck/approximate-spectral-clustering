%embeddings of bipartite vs symmetric Laplacian, 20news

clear;
load('20newsstem');
idx = [find(gnd == 2);find(gnd==3)];
gnd1 = gnd(idx,:);
% fea = min(fea, 10);
fea = tfidf(fea);
fea = fea ./ sqrt(sum(fea.^2,2));
[n,m] = size(fea);
[L,D1,D2] = getLaplacian(fea, 1e-100, 'bipartite');
[U,S,V] = svds(L,20);
figure(1);

%biparite
for t = 0
 if t == 0
  U = D1 * U;
  V = D2 * V;
elseif t > 0
  U = D1*U*S.^t;
  V = D2*V*S.^t;
 end
U(:,1) = [];
V(:,1) = [];
U=U./sqrt(sum(U.^2,2));
V=V./sqrt(sum(V.^2,2));
% ac=embedcluster(U,V,20,gnd,'direct','kmeans')
U = U(idx,:);
subplot(121)
scatter3(U(:,10), U(:,11), U(:,12), 3, gnd1)
% axis([-10e-4 10e-4 -10e-4 10e-4]);
drawnow
pause(1)
end

%symmetric (fast)
d1 = fea * sum(fea, 1)';
t=mean(d1,1);
d1 = max(d1, 1e-100)+t;
D1 = sparse(1:n,1:n,d1.^(-0.5));
w = D1*fea;
[U,S,~]=svds(w,20);

for t = 0
 if t == 0
  U = D1*U;
 elseif t > 0
  U=D1*U*S.^t;
 end
 U(:,1)=[];
 U=U./sqrt(sum(U.^2,2));
%  ac=embedcluster(U,V,20,gnd,'direct','kmeans')
U = U(idx,:);
subplot(122);
scatter3(U(:,1), U(:,2), U(:,3), 3, gnd1)
% axis([-10e-4 10e-4 -10e-4 10e-4]);
drawnow
pause(1)
end

function ac=embedcluster(U,V,k,gnd,embed_method,cluster_method)

if strcmp(embed_method, 'coclustering')
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
    
elseif strcmp(embed_method,'direct')
    label = litekmeans(U, k, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates',10);
    label=bestMap(gnd,label);
    ac=sum(label==gnd)/length(gnd);
end
end

