% Preparing environments
addpath(genpath('/home/drproduck/Documents/MATLAB/SPECTRAL_CLUSTERING/'));

%% Visualizing intrinsic geometry of embedding points
% Summary of example objective
a = [1,2]; b = [5,6];
A = mvnrnd(a, [1,0;0,2], 200);
B = mvnrnd(b, [2,0;0,1], 200);
figure(1)
scatter(A(:,1), A(:,2), [], [1,0,0])
hold on
scatter(B(:,1), B(:,2), [], [0,1,0])
hold off
fea = [A;B];
reps = [a;b];


%% Section 1 Title
% Calculate eigenvectors
W = EuDist2(fea, fea, false);
[n,m] = size(W);
sigma = 2;
W = exp(-W / (2 * sigma^2));
d1 = sum(W, 2);
d1 = max(d1, 1e-100);
d2 = sum(W, 1);
d2 = max(d2, 1e-100);

D1 = sparse(1:n, 1:n, d1.^(-0.5));
D2 = sparse(1:m, 1:m, d2.^(-0.5));

L = D1 * W * D2;
[U,S,V] = svds(L, 2);
figure(2)
subplot(211)
scatter(U(:,1), U(:,2))


%% Section 2 Title
% landmark embedding
W2 = EuDist2(fea, reps, false);
[n,m] = size(W2);
W2 = exp(-W2 / (2 * sigma^2));
d1 = sum(W2, 2);
d1 = max(d1, 1e-100);
d2 = sum(W2, 1);
d2 = max(d2, 1e-100);

D1 = sparse(1:n, 1:n, d1.^(-0.5));
D2 = sparse(1:m, 1:m, d2.^(-0.5));

L2 = D1 * W2 * D2;
[U2,S2,V2] = svds(L2, 2);
subplot(212)
scatter(U2(:,1), U2(:,2))

%% A non-convex dataset
load('circledata_50');
W3 = EuDist2(fea,fea,false);
W3 = exp(-W3 / (2 * sigma^2));
figure(3)
subplot(211)
heatmap(W3);
[n,m] = size(W3);
d1 = sum(W3, 2);
d1 = max(d1, 1e-100);
d2 = sum(W3, 1);
d2 = max(d2, 1e-100);

D1 = sparse(1:n, 1:n, d1.^(-0.5));
D2 = sparse(1:m, 1:m, d2.^(-0.5));

L3 = D1 * W3 * D2;
subplot(212)
heatmap(L3);
