load('20NewsHome.mat')
f=fopen('vocabulary.txt');
vocab=textscan(f,'%s');
vocab=vocab{1};
save('20newsorigin','fea','gnd','vocab');