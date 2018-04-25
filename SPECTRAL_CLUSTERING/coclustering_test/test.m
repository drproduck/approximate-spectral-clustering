load('news3');
issparse(fea)

n = size(fea,1);
m = size(fea,2);
d1 = sum(fea, 2);
d1 = max(d1, 1e-100);
d2 = sum(fea, 1);
d2 = max(d2, 1e-100);

D1 = sparse(1:n, 1:n, d1.^(-0.5));
D2 = sparse(1:m, 1:m, d2.^(-0.5));

L = D1 * fea * D2;
issparse(L);
tic;
[U,S,V] = svds(L, 4);
S

U = D1 * U;
V = D2 * V;
U(:,1) = [];
V(:,1) = [];

U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
V = bsxfun(@rdivide, V, sqrt(sum(V.^2, 2)));
% scatter(U(:,1), U(:,2), 15, gnd);
% scatter3d(U, gnd);
color = zeros(size(gnd, 1), 3);
for i = find(gnd == 1)
    color(i, 1) = 1;
end
for i = find(gnd == 2)
    color(i, 2) = 1;
end
for i = find(gnd == 3)
    color(i, 3) = 1;
end
for i = find(gnd == 4)
    color(i, [1,2]) = 1;
end
U = U(1:1000, :);
color = color(1:1000,:);
scatter3(U(:,1), U(:,2), U(:,3), 50, color);
hold on
scatter3(V(:,1), V(:,2), V(:,3), 15);
toc