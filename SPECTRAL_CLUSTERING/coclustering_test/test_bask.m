clear;
addpath(genpath('/srv/home/kpham/approximate-spectral-clustering/SPECTRAL_CLUSTERING/'));
load('20newsnonstop.mat', 'fea', 'gnd');
nlabel = max(gnd);
%% doc_term
m = size(fea,2);
%ix = 0.01:0.01:0.1;
%i = 1;
ac2 = zeros(60,2);
for it = 1:60
fea = reduce2(fea, m-100*it, 'keep', false);
for i = 1:2
%for remove_low = ix
    %fprintf('remove %.2f%% lowest points\n',remove_low*100)
    fprintf('reducing matrx to %d least common words, try %d\n',m-100*it, i);
    fprintf('sanity check: no words is %d\n', size(fea,2));    
    z_row = find(sum(fea, 2) == 0);
    if max(size(z_row)) ~= 0
	fea(z_row,:) = [];
	fprintf('null row(s) exist. no documents after removed is %d\n', size(fea,1));
    end
    ac2(it,i) = doc_term(fea, nlabel,gnd,1);
end
fprintf('average accuracy is %.2f\n', mean(ac2(it,:), 2));
clear fea;
load('20newsnonstop.mat','fea');
end
save('accuracy_subset_docterm_reverse', 'ac2');
%plot(ix, ac);


%% fast cosine + outlier removal
% ac = zeros(10,1);
% ix = 0.01:0.01:0.1;
% i = 1;
% for remove_low = ix
%     fprintf('remove %.2f%% lowest points\n',remove_low*100)
%     ac(i) = cosine(fea, nlabel,gnd,remove_low);
%     i = i + 1;
% end
% plot(ix, ac);
%% return highest word frequencies from feature matrix
% [~,~,word] = wf(fea,gnd, 5, false, vocab)

%% kmeans with landmark and outlier removal
% ac = zeros(10,1);
% ix = 0.01:0.01:0.1;
% i = 1;
% for remove_low = ix
%     fprintf('remove %.2f%% lowest points\n',remove_low*100)
%     ac(i) = landmark(fea, nlabel,gnd,remove_low);
%     i = i + 1;
% end
% plot(ix, ac);

%SUBFUNCTIONS

%% kmeans landmark + outlier removal
function ac = landmark(fea,nlabel, gnd,remove_low)
[label, kept_idx] = BASK(fea, nlabel, 500,5,2, 'cosine', 'remove_outlier',...
    [0,remove_low],'select_method','kmeans', 'embed_method', 'direct');
gnd = gnd(kept_idx);
label = bestMap(gnd, label);
ac = sum(label == gnd) / size(gnd, 1)
end

%% doc_term + outlier removal
function ac = doc_term(fea,nlabel,gnd,remove_low)
[label,kept_idx] = fast_cosine_SC(fea, nlabel, 2,'cluster_method','kmeans','remove_low',remove_low,'embed_method','coclustering',...
    'remove_high',1,'affinity_type','doc_term');
gnd = gnd(kept_idx);
label = bestMap(gnd, label);
ac = sum(label == gnd) / size(gnd, 1)
end

%% fast cosine + outlier removal
function ac = cosine(fea, nlabel, gnd,remove_low)
[label,kept_idx] = fast_cosine_SC(fea, nlabel, 0,'cluster_method','kmeans','remove_low',remove_low,...
    'embed_method','direct','remove_high',1,'affinity_type','point');
gnd = gnd(kept_idx);
size(label)
label = bestMap(gnd, label);
ac = sum(label == gnd) / size(gnd, 1)
end

%% make tfidf matrix with number of words reduced
function fea = reduce(fea, no_w, mode, bol_save)
s_col = sum(fea, 1);

[~, idx] = sort(s_col, 'ascend');
idx = idx(1:no_w);
if strcmp(mode, 'keep')
    fprintf('keep %d most common words\n',no_w)
    fea = fea(:,idx);
elseif strcmp(mode, 'remove')
    fprintf('remove %d most common words\n', no_w)
    fea(:,idx) = [];
end
fea = tfidf(fea);
if bol_save
    save(strcat('temp', num2str(no_w)), 'fea')
end
end

%binary version
function fea = reduce2(fea, no_w, mode, bol_save)
%s_col = sum(fea, 1);
s_col = sum(fea ~= 0,1);

[~, idx] = sort(s_col, 'ascend');
idx = idx(1:no_w);
if strcmp(mode, 'keep')
    fprintf('keep %d most common words\n',no_w)
    fea = fea(:,idx);
elseif strcmp(mode, 'remove')
    fprintf('remove %d most common words\n', no_w)
    fea(:,idx) = [];
end
fea = tfidf(fea);
if bol_save
    save(strcat('temp', num2str(no_w)), 'fea')
end
end
%% subfunction to calculate cumulative word frequency and return highest ones
function [inds, val, word] = wf(fea, gnd, d, bnormalize, vocab)
%d: number of highest words to return
%return: size(gnd,1) * d matrix. Each row contains indices of highest frequency
%words that appear in the cluster
if bnormalize
    fea = fea ./ sqrt(sum(fea.^2, 2));
end
nlabel = max(gnd);
inds = zeros(nlabel, d);
val = zeros(nlabel, d);
word = cell(nlabel, d);
for i = 1:nlabel
    fprintf('processing group %d...\n',i)
    fre = sum(fea(gnd == i, :), 1);
    [sortfre, idx] = sort(fre, 'descend');
    inds(1,:) = idx(1,1:d);
    val(1,:) = sortfre(1,1:d);
    word(i,:) = vocab(inds(1,:));
end
end
