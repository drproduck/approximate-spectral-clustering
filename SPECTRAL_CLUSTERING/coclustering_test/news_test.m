clear;
load('20news10879');
idx = [find(gnd == 4);find(gnd==3)];
gnd = gnd(idx,:);
fea = min(fea, 10);
fea = tfidf(fea);
fea = fea ./ sqrt(sum(fea.^2,2));
[n,m] = size(fea);
[L,D1,D2] = getLaplacian(fea, 1e-100, 'bipartite');
[u,s,v] = svds(L,20);

figure(1);
for t = 0
 if t == 0
  U = D1 * u;
  V = D2 * v;
 elseif t > 0
  U = D1*u*s.^t;
  V = D2*v*s.^t;
 end
 U(:,1) = [];
 V(:,1) = [];
U = U(idx,:);
subplot(121)
scatter3(U(:,10), U(:,11), U(:,12), 3, gnd)
% axis([-10e-4 10e-4 -10e-4 10e-4]);
drawnow
pause(1)
end

d1 = fea * sum(fea, 1)';
d1 = max(d1, 1e-100);
D1 = sparse(1:n,1:n,d1.^(-0.5));
w = D1*fea;
[u,s,v]=svds(w,20);

for t = 0
 if t == 0
  U = D1 * u;
 elseif t > 0
  U = D1*u*s.^t;
 end
 U(:,1) = [];
U = U(idx,:);
subplot(122);
scatter3(U(:,10), U(:,11), U(:,12), 3, gnd)
% axis([-10e-4 10e-4 -10e-4 10e-4]);
drawnow
pause(1)
end

