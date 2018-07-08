function [center_ind] = D2sample(fea, r)
%D2SAMPLE Sample points evenly on the dataset according to kmeans++
%weighting
%   fea: row-major dataset
%   r: number of sampled points

n = size(fea, 1);
center_ind = zeros(r,1);
ind = randsample(n,1);
center_ind(1) = ind;
closest_distance = EuDist2(fea, fea(ind,:), false);
closest_distance(ind) = 0;
for i = 2:r
    ind = randsample(n, 1, true, closest_distance);
    center_ind(i) = ind;
    distance_to_new_center = EuDist2(fea, fea(ind,:), false);
    closest_distance = min(closest_distance, distance_to_new_center);
    closest_distance(ind) = 0;
end



