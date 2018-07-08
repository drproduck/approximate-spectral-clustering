clear;
maxt = 10;
maxit = 2;
seed = 99999;

bask_test('letter', 'gaussian', seed, 500, 5, maxt, maxit)
bask_test('mnist', 'gaussian', seed, 500, 5, maxt,maxit)
bask_test('pend', 'gaussian',seed, 500, 5, maxt,maxit)
bask_test('usps', 'gaussian',seed, 500, 5, maxt,maxit)
bask_test('shuttle', 'gaussian',seed, 500, 5, maxt,maxit)
bask_test('protein', 'gaussian',seed, 500, 5, maxt,maxit)

% bask_test('musk_1', 'gaussian',seed, 500, 5, maxt,maxit)
% bask_test('connect4_binary_1', 'gaussian',seed, 500, 5, maxt,maxit)
% bask_test('poker', 'gaussian',seed, 500, 5, maxt,maxit)

% bask_test('news', 'cosine',seed, 500, 5, maxt,maxit)
% bask_test('TDT2','cosine',seed,500,5,maxt,maxit)
% bask_test('Reuters21578', 'cosine', seed,500,5,maxt,maxit)
% bask_test('TDT2', 'cosine',seed,500,5,maxt,maxit)



