clear;
initRes = 10;
initIter = 100;
seed = 99999;

mat = {'letter' 'mnist' 'usps' 'protein' 'pend' 'shuttle'};

for i = 1:6
   fprintf('Running kmeans on %s\n', mat{i}); 
   load(mat{i});
   nlabel = max(gnd);
   t0 = cputime;
   labels = litekmeans(fea, nlabel, 'replicates', initRes, 'maxiter', initIter, 'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100, 'clustersample', 0.1);
   t = cputime - t0;
   labels = bestMap(gnd, labels);
   acc = sum(labels == gnd) / size(gnd, 1) * 100;
   fprintf('accuracy = %f\n', acc)
   fprintf('time = %f\n', t)
end