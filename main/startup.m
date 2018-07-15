%fire things up
addpath(genpath('/home/drproduck/approximate-spectral-clustering/'));
ok = true;
if max(size(strfind(path, 'SPECTRAL_CLUSTERING'))) == 0
    ok = false;
end
    
if ok
    disp('startup ok')
else 
    disp('something may be wrong when starting up')
end