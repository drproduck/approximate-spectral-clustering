% This spectral clustering code requires Koutis's linear solver 
% which is available at: http://ccom.uprrp.edu/~ikoutis/cmg.html
% we can use directly from knn graph or from feature data

% If the knn graph is given, we can just use the fucntion
% C = kmeans_CTD_ST(G, k, nInits, maxIter, kRP, scale);  

% If the knn graph is not given, we use the function
% C = resistance_spectral_clustering(X,k,k1,kRP)
% For knn graph: we need to use the following external library
% k-d tree: https://sites.google.com/site/andreatagliasacchi/software/matlabkd-treelibrary
% Graph library: http://dgleich.github.io/matlab-bgl/

function C = resistance_spectral_clustering(X,k,k1,kRP)
% spectral clustering using resistance distance
% X: data set (objects x features)
% k: number of clusters
% k1: number of k nearest neighbours of graph (e.g. k1 = 10)
% kRP: columns of random projection matrix (e.g. kRP = 200)
% C: cluster membership

% X = csvread('Datasets/testCDST3.txt');
% k = 6;
% k1 = 10;
% kRP = 50;

% parameters for knn graph
graph_type = 1; % knn graph type
similarity = 'euclidean'; % knn similarity metric
sigma = 0;

% parameters for k-means
nInits = 100;     % number of initialisation
maxIter = 100;   % maximum number of iterations

scale = 1;       % sparsity of the random matrix

if sigma == 0 && strcmp(similarity, 'RBF') % self tune
    sigma = kernel_bandwidth(X, k1);
end

G = knn_components_sparse(X, k1, graph_type, similarity, sigma);
% if the data is big, use the following function for faster creation of knn graph
% G = knn_components_sparse_RP(X, kRP, k1, 2*k1, graph_type, similarity, sigma, scale);

G = connect_components_mix(X, G, similarity, sigma);

% if the graph is given, we can just use the following function
C = kmeans_CTD_ST(G, k, nInits, maxIter, kRP, scale);    

plotClusters('RESC', X, C, [], false);


%------------------------------------------------------------------
function [ids, Y, Z_time, kmeans_time, total_time] ... 
    = kmeans_CTD_ST(A, k, nInits, maxIter, kRP, scale)
% k-means clustering using approximate resistance embedding, 
% with multiple centroids initialization
% A: graph adjacency matrix
% k: # of clusters
% nInits: # of centroid initialization
% maxIter: maximum number of iterations
% kRP: # columns of random projection matrix (kRP = O(logn))
% scale: sparsity of the random matrix (e.g. scale=1)
% ids: cluster membership for each data point
% Y: eigenspace
% Z_time, kmeans_time, total_time: time for computation

t = tic;

tZ = tic;
disp 'Build matrix Z using Spielman-Tang solver ...'
Z = approx_resistance(A, kRP, 1e-6, 100, scale);
Y = Z';
clear Z;
Z_time = toc(tZ);

% clustering in the Laplacian eigenspace
tkmeans = tic;
disp 'kmeans in Z space ...'
ids = kmeans(Y, k, 'start', 'cluster', 'emptyaction', 'singleton', ...
	 'onlinephase', 'off', 'replicates', nInits, 'maxiter', maxIter, 'display', 'final');
kmeans_time = toc(tkmeans);
	
total_time = toc(t);


%------------------------------------------------------------------
function [Z, solver_time] = approx_resistance(A, k, tolerance, maxIter, scale)
% From the n x n adjacency matrix A, create the k x n matrix (k=O(logn))
% Z from which we can calculate the approximate effective resistances 
% between any two nodes in the graph.
% Creation of Z requires solving k linear equations which solve by 
% a linear solver.

% tolerance 	: tolerance of the method
% maxIter   	: maximum number of iterations
% scale     	: decide the sparsity of the random matrix or the scale of the speed up 
%             		the lager the scale, the sparser the random matrix
% solver_time	: the time for calling the linear solver 
n = size(A,1);
[x,y,w] = find(A); % nonzero entries of A
idx = (x < y); % to avoid counting each edge twice
m = nnz(idx);
l = 1:m;
i = [l'; l'];
j = [x(idx); y(idx)];
s = [ones(m,1); -ones(m,1)];
B = sparse(i,j,s,m,n);
W = sparse(1:m,1:m,w(idx),m,m);
D = sparse(1:n,1:n,sum(A,1),n,n);
clear x y w idx l i j s;

L = D - A;
X = (W.^(0.5))*B;   
clear B W D A;

if isempty(scale)
    scale = 1;
end

if scale == 0       % normal distribution
    Y = (randn(k,m)*X)./sqrt(k);
elseif scale == 1   % -1 and 1 with probability 1/2
    Y = (randsrc(k,m)*X)./sqrt(k);
elseif scale == 3   % -sqrt(3) 0 and sqrt(3) with probabilities 1/6, 2/3, and 1/6
    Y = (randsrc(k,m,[-1 0 1; 1/6 2/3 1/6])*X).*(sqrt(3)/sqrt(k));
else                % -sqrt(s) 0 and sqrt(s) with probabilities 1/2s, 1-1/s, and 1/2s
    Y = (randsrc(k,m,[-1 0 1; 1/(2*scale) 1-1/scale 1/(2*scale)])*X).*(sqrt(scale)/sqrt(k));
end
    
clear X;

tic
Z = zeros(k,n);
pfun = cmg_sdd(L);    

for i=1:k
    Z(i,:) = pcg(L, Y(i,:)', tolerance, maxIter, pfun);
end    

solver_time = toc;


%------------------------------------------------------------------
function [sigma, bandwidth] = kernel_bandwidth(X, k)
% find the bandwith of each data point
% help to determine the kernel bandwidth

% X: dataset
% k: knn

n = size(X,1); % No. of observations
   
bandwidth = zeros(n,1); % bandwidth for each data point

% build a kd tree
tree = kdtree_build(X);

for i=1:n   
    % knn
    q = X(i,:);    
    % the nearest point (the same point) is at the end, just ignore it 
    inn = kdtree_k_nearest_neighbors(tree, q, k+1);    % in euclidean
    
    bandwidth(i) = dist(X(i,:),X(inn(1),:)); % distance to k-th nearest neighbor
end

sigma = max(bandwidth);
kdtree_delete(tree);


%------------------------------------------------------------------
function [A, tree] = knn_components_sparse(X, k, graph_type, similarity, sigma)
% optimize the sparse matrix
% find a k-nn graph of the dataset X, can be disconnected
% graph_type = 1: knn
% graph_type = 2: mutual knn
% graph_type = 3: fully connected graph
% similarity = how to compute weights of the edge
% A: graph adjacency matrix

if nargin<5
    sigma = [];
end

n = size(X,1); % No. of observations
   
% fully connected graph
if graph_type == 3
    A = zeros(n,n); % dense graph
    for i=1:n
        for j=i+1:n
            w = sim(X(i,:), X(j,:), similarity, sigma);
            if w > 0
                A(i,j) = w;
                A(j,i) = w;
            end
        end
    end
    
    A = sparse(A); % sparse graph
    return
end

% nearest neighbor graph
nodes1 = zeros(2*n*k,1); % nk = maximum number of nnz for knn graph
nodes2 = nodes1;
nodeweights = nodes1;
ne = 0;

% build a kd tree
tree = kdtree_build(X);

for i=1:n   
    % knn
    q = X(i,:);    
    % the nearest point (the same point) is at the end, just ignore it 
    inn = kdtree_k_nearest_neighbors(tree, q, k+1);    % in euclidean
    
    for l=1:k
        w = sim(X(i,:), X(inn(l),:), similarity, sigma);
        
        if w < 1e-6 % not suitable sigma
            sprintf('sigma is too small');
            w = 1e-6;
        end
        
        ne = ne + 1;
        nodes1(ne) = i;
        nodes2(ne) = inn(l);
        nodeweights(ne) = w;
        
        if graph_type == 1
            ne = ne + 1;        
            nodes1(ne) = inn(l);
            nodes2(ne) = i;
            nodeweights(ne) = w;                    
        end
    end
end

if graph_type == 2 % mutual knn graph
    nodes1(ne+1:2*n*k) = [];
    nodes2(ne+1:2*n*k) = [];
    nodeweights(ne+1:2*n*k) = [];

    for i=1:ne
        neighbors = nodes1==nodes2(i);
        flag = find(nodes2(neighbors)==nodes1(i),1);
        if isempty(flag) % not found
            nodeweights(i) = 0;  
        end
    end
end

% remove duplicate information in nodes1 and nodes2 to avoid adding in
% nodeweights by sparse function
[nodes,index] = unique([nodes1 nodes2],'rows','first');
nodeweights = nodeweights(index);

A = sparse(nodes(:,1), nodes(:,2), nodeweights, n, n);

if nargout < 2
    kdtree_delete(tree);
end


%------------------------------------------------------------------
function Anew = connect_components_mix(X, A, similarity, sigma)
% connect disconnected graph components from A by using MST
% for all representative points from each components

% connected components
[ci, sizes] = components(A);

Anew = A;
nClusters = size(sizes,1);
sprintf('There are %d disconnected components', nClusters)
if nClusters > 1 % disconnected
    indexT = zeros(nClusters,1); % store the representative point from all components
        
    for i=1:nClusters
        index = find(ci' == i,1); % any point from the cluster
        indexT(i) = index;
    end

    tree = connect_components_MST(X(indexT,:), A(indexT,indexT), similarity, sigma);

    Anew(indexT,indexT) = tree;
end


%------------------------------------------------------------------
function Anew = connect_components_MST(X, A, similarity, sigma)
% connect disconnected graph components from using inverse MST
% this MST tends to connect nodes which near to each other
% so that there will be more overlapping between MST and A

Anew = A;

E = euclidean(X);
T = mst(sparse(E));
% T = affinity(X, similarity, sigma);

[i,j] = find(T);
for l=1:nnz(T)
    if i(l)<j(l) && A(i(l),j(l))==0
        s = sim_d(T(i(l),j(l)), similarity, sigma);
        if s < 1e-6 % not suitable sigma
            sprintf('sigma is too small');
            s = 1e-6;
        end
        Anew(i(l),j(l)) = s;
        Anew(j(l),i(l)) = s;
    end
end


%------------------------------------------------------------------
function E = euclidean(X)
% compute the euclidean distances
n = size(X,1); % observation

E = zeros(n,n);
for i=1:n
    for j=i+1:n
        E(i,j) = dist(X(i,:),X(j,:));
        E(j,i) = E(i,j);
    end
end


%------------------------------------------------------------------
function si = sim(x, y, s, sigma)
% compute the similarity b/w two observations x and y
% sigma: bandwidth for RBF function
% s: similarity type

if strcmp(s,'cosine') == 1
    si = cosine(x,y);    
elseif strcmp(s,'RBF') == 1
    si = RBF(x,y,sigma);   
else
    d = dist(x,y);
    if d~=0
        si = 1/d;
    else
        si = 0;
    end
end


%------------------------------------------------------------------
function si = sim_d(d, s, sigma)
% compute the similarity b/w two observations with euclidean distance d
% s: similarity type
% only use for euclidean and RBF

if strcmp(s,'RBF') == 1
    si = exp(-d^2/(2*sigma*sigma));
else
    if d~=0
        si = 1/d;
    else
        si = 0;
    end
end


%------------------------------------------------------------------
function d = dist(x, y)
% compute the euclidean distance b/w two observations
d = norm(x - y);


%------------------------------------------------------------------
function c = cosine(x, y)
% compute the cosine between two vectors
c = dot(x,y)/(norm(x)*norm(y));


%------------------------------------------------------------------
function s = RBF(x, y, sigma)
% compute the RBF function
d = dist(x, y);
s = exp(-d^2/(2*sigma*sigma));


%------------------------------------------------------------------
function plotClusters(name, X, C, M, plotCenters)
% plot observations in X based on the clusters in C
% plot centroids in M with bigger size if plotCenters = true
% name: name of the figure

figure('Name',name, 'NumberTitle','off');

n = size(C, 1);
d = size(X,2);
k = size(M, 1);
hold on;
%axis([0, 100, 0, 100]);
for i=1:n
    if C(i)==1
        color = '+r';
    elseif C(i)==2
        color = 'og';
    elseif C(i)==3
        color = '*b';
    elseif C(i)==4
        color = 'xc';    
    elseif C(i)==5
        color = 'sm';    
    elseif C(i)==6
        color = 'dy';    
    elseif C(i)==7
        color = '^k';      
    elseif C(i)==8
        color = 'vr';   
    elseif C(i)==9
        color = 'pg';   
    else 
        color = 'hb';   
    end
    if d>=3
        h = plot3(X(i,1), X(i,2), X(i,3), color);
    else
        h = plot(X(i,1), X(i,2), color);
    end
end

if plotCenters == true
    for i=1:k
        if k>7
            color = '.r';
        elseif i==1
            color = '+r';
        elseif i==2
            color = 'og';
        elseif i==3
            color = '*b';
        elseif i==4
            color = 'xc';    
        elseif i==5
            color = 'sm';    
        elseif i==6
            color = 'dy';    
        elseif i==7
        color = '^k';      
        elseif i==8
            color = 'vr';   
        elseif i==9
            color = 'pg';   
        else 
            color = 'hb';   
        end
        if size(M,2) == 1 % centroids are point index in the cluster
            if d>=3
                h = plot3(X(M(i),1), X(M(i),2), X(M(i),3), color, 'MarkerSize', 10, 'LineWidth', 3);
            else
                h = plot(X(M(i),1), X(M(i),2), color, 'MarkerSize', 10, 'LineWidth', 3);
            end
        else % centroids are not points in the cluster
            if d>=3
                h = plot3(M(i,1), M(i,2), M(i,3), color, 'MarkerSize', 10, 'LineWidth', 3);
            else
                h = plot(M(i,1), M(i,2), color, 'MarkerSize', 10, 'LineWidth', 3);
            end
        end
    end
end

hold off;


%------------------------------------------------------------------
function [A, tree] = knn_components_sparse_RP(X, kRP, k, h, graph_type, similarity, sigma, scale)
% optimize the sparse matrix and do the random projection
% find a k-nn graph of the dataset X, can be disconnected
% kRP: random projection parameter
% k: the number of knn in the original space
% h: the number of knn in the random projection space (h=2*k is OK)
% scale: sparsity for random projection
% graph_type = 1: knn
% graph_type = 2: mutual knn
% graph_type = 3: fully connected graph
% similarity = how to compute weights of the edge
% A: graph adjacency matrix

if isempty(scale)
    scale = 1;
end

n = size(X,1); % No. of observations

Y = randomprojection(X, kRP, scale);

% fully connected graph
if graph_type == 3
    A = zeros(n,n); % dense graph
    for i=1:n
        for j=i+1:n
            w = sim(Y(i,:), Y(j,:), similarity, sigma);
            if w > 0
                A(i,j) = w;
                A(j,i) = w;
            end
        end
    end
    
    A = sparse(A); % sparse graph
    return
end

% nearest neighbor graph
nodes1 = zeros(2*n*k,1); % nk = maximum number of nnz for knn graph
nodes2 = nodes1;
nodeweights = nodes1;
ne = 0;

% build a kd tree
tree = kdtree_build(Y);

for i=1:n   
    % find h nearest neighbors in the projected space
    q = Y(i,:);    
    % the nearest point (the same point) is at the end, just ignore it 
    inn = kdtree_k_nearest_neighbors(tree, q, h+1);    
    
    % find the knn in the original space
    idx = knnsearch(X(inn(1:h),:),X(i,:),'NSMethod','exhaustive','k',k);     
    for l=1:k
        w = sim(X(inn(idx(l)),:), X(i,:), similarity, sigma);
        
        if w < 1e-6 % not suitable sigma
            sprintf('sigma is too small');
            w = 1e-6;
        end
        
        ne = ne + 1;
        nodes1(ne) = i;
        nodes2(ne) = inn(idx(l));
        nodeweights(ne) = w;
        
        if graph_type == 1
            ne = ne + 1;        
            nodes1(ne) = inn(idx(l));
            nodes2(ne) = i;
            nodeweights(ne) = w;                    
        end
    end
end

if graph_type == 2 % mutual knn graph
    nodes1(ne+1:2*n*k) = [];
    nodes2(ne+1:2*n*k) = [];
    nodeweights(ne+1:2*n*k) = [];

    for i=1:ne
        neighbors = nodes1==nodes2(i);
        flag = find(nodes2(neighbors)==nodes1(i),1);
        if isempty(flag) % not found
            nodeweights(i) = 0;  
        end
    end
end

% remove duplicate information in nodes1 and nodes2 to avoid adding in
% nodeweights by sparse function
[nodes,index] = unique([nodes1 nodes2],'rows','first');
nodeweights = nodeweights(index);

A = sparse(nodes(:,1), nodes(:,2), nodeweights, n, n);

if nargout < 2
    kdtree_delete(tree);
end


%------------------------------------------------------------------
function Y = randomprojection(X, kRP, scale)
% randomly project data in X to Y
% X: dataset
% kRP: dimension in the new space (Y)
% scale: projection style

m = size(X,2);
if scale == 0       % normal distribution
    R = randn(m,kRP)./sqrt(kRP);
elseif scale == 1   % -1 and 1 with probability 1/2
    R = randsrc(m,kRP)./sqrt(kRP);
elseif scale == 3   % -sqrt(3) 0 and sqrt(3) with probabilities 1/6, 2/3, and 1/6
    R = randsrc(m,kRP,[-1 0 1; 1/6 2/3 1/6]).*(sqrt(3)/sqrt(kRP));
else                % -sqrt(s) 0 and sqrt(s) with probabilities 1/2s, 1-1/s, and 1/2s
    R = randsrc(m,kRP,[-1 0 1; 1/(2*scale) 1-1/scale 1/(2*scale)]).*(sqrt(scale)/sqrt(kRP));
end
Y = X*R;