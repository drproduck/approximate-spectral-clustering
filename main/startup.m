%fire things up
addpath(genpath('/dfs/scratch0/itsmemario/approximate-spectral-clustering/'));
ok = false;
if strfind(path, 'approximate-spectral-clustering-dummy')
    ok = true;
end
    
if ok
    disp('startup ok')
else 
    disp('something may be wrong when starting up')
end
