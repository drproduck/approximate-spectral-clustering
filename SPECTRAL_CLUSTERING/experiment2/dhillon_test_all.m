clear;
maxt = 20;
maxit = 50;
seed = 99999;

dhillon_test('letter', 'gaussian', seed, 500, 5, maxit)
% bask_test('mnist', 'gaussian', seed, 500, 5, maxt,maxit)
% bask_test('pend', 'gaussian',seed, 500, 5, maxt,maxit)
% bask_test('usps', 'gaussian',seed, 500, 5, maxt,maxit)
% bask_test('shuttle', 'gaussian',seed, 500, 5, maxt,maxit)
% bask_test('musk_1', 'gaussian',seed, 500, 5, maxt,maxit)
% bask_test('connect4_binary_1', 'gaussian',seed, 500, 5, maxt,maxit)
% bask_test('poker', 'gaussian',seed, 500, 5, maxt,maxit)
% bask_test('protein', 'gaussian',seed, 500, 5, maxt,maxit)
% dhillon_test('news100', 'cosine',seed, 500, 5, maxit)