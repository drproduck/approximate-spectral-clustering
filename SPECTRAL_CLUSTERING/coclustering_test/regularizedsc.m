function label = regularizedsc(fea,gnd,nlabel)
[n,m] = size(fea);
fea=tfidf(fea,'hard');
fea=fea./sqrt(sum(fea.^2,2));
[l,d1,d2]=getLaplacian(fea,1e-12,'bipartite');
[u,s,v]=svds(l,nlabel);
u=d1*u;
v=d2*v;
u(:,1)=[];
v(:,1)=[];
u=u./sqrt(sum(u.^2,2));
v=v./sqrt(sum(v.^2,2));
w=[u;v];
[allabel,~] = litekmeans(w, 20, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates',10);
label=allabel(1:n);
label = bestMap(gnd, label);
% ac=sum(doclabel==gnd) / length(gnd)