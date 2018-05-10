load('20newsnonstop.mat');
col_sum = sum(fea ~= 0, 1);
[~, idx] = sort(col_sum, 'descend');
[n,m] = size(fea);
no_word = m - 50000;
kept_idx = idx(1:no_word);
fea = fea(:,kept_idx);
f = fopen('vocab');
vocab = textscan(f, '%s');
vocab = vocab{1};
vocab = vocab(kept_idx);
row_sum = sum(fea ~= 0, 2);
if sum(row_sum == 0) ~= 0
    disp('null rows(s) exist')
end
null_row_idx = find(row_sum == 0);
fea(null_row_idx,:) = [];
gnd(null_row_idx,:) = [];
row_sum = sum(fea ~= 0, 2);
if sum(row_sum == 0) ~= 0
    disp('null row(s) still exist')
end
fprintf('fea size: %d %d\n', size(fea, 1), size(fea, 2))
fprintf('gnd size: %d %d\n', size(gnd, 1), size(gnd, 2))
fprintf('vocab size: %d %d\n', size(vocab, 1), size(vocab, 2))
save('20news10879', 'fea','gnd','vocab')