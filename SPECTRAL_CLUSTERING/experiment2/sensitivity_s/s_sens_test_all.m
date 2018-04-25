clear;
maxt = 1;
maxit = 20;
seed = 99999;
r = 500;
S = [1,1,10];

s_sens_test('letter', 'gaussian', seed, r, S, maxt, maxit)
s_sens_test('mnist', 'gaussian', seed, r, S, maxt,maxit)
s_sens_test('usps', 'gaussian',seed, r, S, maxt,maxit)
s_sens_test('protein', 'gaussian',seed, r, S, maxt,maxit)
% s_sens_test('pend', 'gaussian',seed, r, S, maxt,maxit)
% s_sens_test('shuttle', 'gaussian',seed, r, S, maxt,maxit)
% s_sens_test('musk_1', 'gaussian',seed, r, S, maxt,maxit)
% s_sens_test('connect4_binary_1', 'gaussian',seed, r, S, maxt,maxit)
% s_sens_test('news100', 'cosine',seed, r, S, maxt,maxit)
% s_sens_test('poker', 'gaussian',seed, r, S, maxt,maxit)



