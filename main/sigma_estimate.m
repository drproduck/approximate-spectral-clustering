mat = {'letter','mnist','pend','usps','shuttle','protein'};
sigma = zeros(length(mat), 1);
% estimate with 5000 sampled points
nSampleCap = 5000;
for i = 1:length(mat)
   load(mat{i}, 'fea');
   n = size(fea,1);
   if n <= nSampleCap
       nSample = n
   else
       nSample = nSampleCap
   end
   
   fea = fea(randsample(n, nSample), :);
   size(fea) 
   % 7 knn
   W = EuDist2(fea, fea, 0);
   s = 7;  
   dump = zeros(nSample, s);
   idx = dump;
   for j = 1:s
        [dump(:,j),idx(:,j)] = min(W,[],2);
        temp = (idx(:,j)-1)*nSample+(1:nSample)';
        W(temp) = 1e+100; 
   end
    
    sigma(i) = mean(mean(dump))
    disp(strcat('done ',mat{i}))
end

save('sigma')
