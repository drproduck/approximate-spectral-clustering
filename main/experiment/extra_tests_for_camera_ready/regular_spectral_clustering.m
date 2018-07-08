clear;
seed = 99999;

addpath(genpath('/home/drproduck/Documents/MATLAB/SPECTRAL_CLUSTERING/'))
mat = {'letter' 'usps' 'protein' 'pend'};
sigma = [0.377625, 7.152103, 2.849037, 28.160720];

for i = 1:4
   fprintf('Running spectral clustering on %s\n', mat{i}); 
   load(mat{i});
   nlabel = max(gnd);
   t0 = cputime;
   opts.sigma = sigma(i);
   labels = NCut(fea, nlabel, 'gaussian', opts);
   t = cputime - t0;
   labels = bestMap(gnd, labels);
   acc = sum(labels == gnd) / size(gnd, 1) * 100;
   fprintf('accuracy = %f\n', acc)
   fprintf('time = %f\n', t)
   clear opts;
   clear fea;
   clear gnd;
end