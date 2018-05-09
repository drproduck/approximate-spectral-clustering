load('20NewsHome');
id = fopen('vocabulary.txt');
vocab = textscan(id, '%s');
vocab = vocab{1};
stop_word = importdata('stop_word_inds')
new_vocab = vocab;
new_vocab(stop_word) = [];
size(new_vocab)
fea(:,stop_word) = [];
size(fea);
save('20newsnonstop','fea','gnd')
id = fopen('vocab','w');
for i = 1:size(new_vocab)
    fprintf(id, '%s\n',new_vocab{i});
end