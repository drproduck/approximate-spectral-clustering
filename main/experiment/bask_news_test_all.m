clear;
addpath(genpath('/home/drproduck/Documents/MATLAB/SPECTRAL_CLUSTERING/'));
maxt = 5;
maxit = 1;
seed = 99999;
s = 3;

bask_news_test('news',seed, s, maxt,maxit)
% bask_news_test('tdt',seed, s, maxt,maxit)
% bask_news_test('reuters',seed, s, maxt,maxit)
% bask_news_test('reuters30_new',seed, s, maxt,maxit)