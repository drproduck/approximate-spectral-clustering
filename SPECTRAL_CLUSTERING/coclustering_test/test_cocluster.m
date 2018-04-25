clear;
addpath(genpath('/home/drproduck/Documents/MATLAB/SPECTRAL_CLUSTERING/'));
load('news')
[n,m] = size(fea);
% [U,S,V] = bask_doc_term(fea, 20, 0);
tic;[U,S,V] = save_eigs('news_eigs.mat', fea, 20, 0);toc
load('data_20news');
% W = [U;V];
% group_by_affinity(U,V, 10000, 5, gnd,cls_names, words)
% group_by_cluster(U,V, n, 5, gnd, cls_names, words) 
% lsi(U,S,V, 10000, 5, gnd,cls_names, words)
group_by_affinity_learned(U,V, 9000, 5, gnd,cls_names, words)

function [] = lsi(U,S,V, no_sample, top_k, gnd, cls_names, words)
n = size(U,1);
ind = randsample(n, no_sample);
U = U(ind, :);
gnd = gnd(ind, :);
affinity = U * S * V';

for i = 1:20
    A = affinity(gnd == i, :);
    word_score = sum(A, 1);
    [sc, max_indx] = maxk(word_score, top_k);
    fprintf('group %s, most frequent words: ', cls_names{i})
    for j = 1:top_k
        fprintf('%s (%f), ', words{max_indx(j)}, sc(j))
    end
    fprintf('\n')
end

end
function [] = group_by_affinity(U,V, no_sample, top_k, gnd, cls_names, words)
n = size(U,1);
ind = randsample(n, no_sample);
U = U(ind, :);
gnd = gnd(ind, :);
affinity = U * V';

%in case grouping is known
for i = 1:20
    A = affinity(gnd == i, :);
    word_score = sum(A, 1);
    [sc, max_indx] = maxk(word_score, top_k);
    fprintf('group %s, most frequent words: ', cls_names{i})
    for j = 1:top_k
        fprintf('%s (%f), ', words{max_indx(j)}, sc(j))
    end
    fprintf('\n')
end
end

function [] = group_by_affinity_learned(U,V, no_sample, top_k, gnd, cls_names, words)
n = size(U,1);
ind = randsample(n, no_sample);
U = U(ind, :);
gnd = gnd(ind, :);
affinity = U * V';

fprintf('Clustering word and doc embedding using kmeans...\n')
learned_label = litekmeans(U, 20, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates',10);
learned_label = bestMap(gnd, learned_label);

for i = 1:20
    A = affinity(learned_label == i, :);
    word_score = sum(A, 1);
    [sc, max_indx] = maxk(word_score, top_k);
    fprintf('group %s, most frequent words: ', cls_names{i})
    for j = 1:top_k
        fprintf('%s (%f), ', words{max_indx(j)}, sc(j))
    end
    fprintf('\n')
end
end

function [] = group_by_cluster(U,V, no_sample, top_k, gnd, cls_names, words)
n = size(U,1);
m = size(V,1);
if no_sample < n
    fprintf('sampling a subset of doc...\n')
    ind = randsample(n, no_sample);
    U = U(ind, :);
    gnd = gnd(ind, :);
end

W = [U;V];
fprintf('Clustering word and doc embedding using kmeans...\n')
[all_label, center] = litekmeans(W, 20, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates',10);
for i = 1:20
    x = hist(gnd(all_label(1:n) == i), 1:20);
    [n_doc, group_idx] = maxk(x, 3);
    doc = U(all_label(1:n) == i, :);
    word = V(all_label(n+1:n+m) == i, :);
    dist_to_center = word * center(i,:)';
    [sc, max_indx] = maxk(dist_to_center, top_k);
    fprintf('group %d with top 3 topics: %s (%d), %s(%d), %s(%d), most frequent words: ', i, ...
        cls_names{group_idx(1)}, n_doc(1), cls_names{group_idx(2)}, n_doc(2), cls_names{group_idx(3)}, n_doc(3))
    for j = 1:top_k
        fprintf('%s (%f), ', words{max_indx(j)}, sc(j))
    end
    fprintf('\n')
end
end

function [] = group_by_nearby_doc(U,V,no_sample,top_k,gnd, cls_names,words)
n = size(U,1);
m = size(V,1);
if no_sample < n
    fprintf('sampling a subset of doc...\n')
    ind = randsample(n, no_sample);
    U = U(ind, :);
    gnd = gnd(ind, :);
end
end




% tsne_plot(U, gnd, 20, 29, 30);
% tnse_plot(W, doc_term_color(gnd, m), 2, 29, 30);
% all_label = litekmeans(W, 3, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates',10);
% scatter(W(:,1), W(:,2), [], doc_term_color(gnd, m))
% save('result_reuter', 'U','V','S','all_label')

function mappedX = tsne_plot(fea, gnd, no_dim, reduced_dim, perp)
mappedX = tsne(fea, gnd, no_dim, reduced_dim, perp);
gscatter(mappedX(:,1), mappedX(:,2), train_labels);
end

function color = doc_term_color(gnd, n_term)
color = [gnd;zeros(n_term, 1)+max(gnd) + 1];
end

function [U,S,V] = save_eigs(mat_name, fea, neigs,t)
if ~exist(mat_name, 'file')
    fprintf('matrix does not exist, processing and returning...\n')
    [U,S,V] = bask_doc_term(fea, neigs, t);
    save(mat_name, 'U', 'S', 'V');
else
    fprintf('maxtrix exists, loading...\n')
    f = matfile(mat_name);
    U = f.U;S = f.S;V = f.V;
end
end

function [val, idx] = maxk(arr, k)
%return k largest elements from array arr
n = size(arr,1);
if k > 5 * log(n)
    [val, idx] = sort(arr, 'descend');
    val = val(1:k);
    idx = idx(1:k);
else
    val = zeros(k,1);
    idx = zeros(k,1);
    for i = 1:k
        [val(i), idx(i)] = max(arr);
        arr(idx(i)) = 1e-100;
    end
end
end
        
    
