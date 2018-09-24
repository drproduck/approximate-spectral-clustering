function reps = reps_by_class(x, labels, r)
nlabel = max(labels);
for i = 1:nlabel
    x_class = x(find(labels==i), :);
    [~, reps_class] = litekmeans(x_class, r, 'MaxIter', 10, 'Replicates', 1,...
            'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100, 'clustersample', 0.1);
    if i == 1
        reps = reps_class;
    else
        reps = [reps;reps_class];
    end
end