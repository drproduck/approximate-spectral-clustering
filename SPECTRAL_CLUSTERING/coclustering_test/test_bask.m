clear;
load('news');
nlabel = max(gnd);
opts.t = 1;
label = BASK(fea, nlabel, 500,5,'cosine', 'kmeans', 'coclustering',opts);
label = bestMap(gnd, label);
sum(label == gnd) / size(gnd, 1)