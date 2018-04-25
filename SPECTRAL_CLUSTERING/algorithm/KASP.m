function [label, X] = KASP(fea, k, r, opts)

if (~exist('opts','var'))
   opts = [];
end

% Testing option
if isfield(opts, 'reps')
    reps = opts.reps;
    label = opts.pre_label;
else
    initIter = 10;
    if isfield(opts, 'initIter')
        initIter = opts.initIter;
    end
    initRep = 1;
    if isfield(opts, 'initRep')
        initRep = opts.initRep;
    end

    [label, reps, ~, VAR] = litekmeans(fea, r, 'MaxIter', initIter, 'Replicates', initRep);
    lbcount = hist(label, 1:r);
end

finalIter = 100;
if isfield(opts, 'finalIter')
    finalIter = opts.finalIter;
end
finalRep = 10;
if isfield(opts, 'finalRep')
    finalRep = opts.finalRep;
end
opts2.finalRep = finalRep;
opts2.finalIter = finalIter;

if isfield(opts, 'sigma')
    opts2.sigma = opts.sigma;
else
    sigma = mean(sqrt(VAR ./ lbcount));
    opts2.sigma = sigma;
end
if isfield(opts, 'sparse')
    opts2.sparse = opts.sparse;
end

[rep_label, X] = SC(reps, k, 'gaussian', opts2);

for i = 1 : size(fea, 1)
    label(i) = rep_label(label(i));
end


