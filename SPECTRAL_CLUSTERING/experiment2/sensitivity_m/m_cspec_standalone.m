clear;
mat = {'usps','letter','protein','mnist'};
maxit = 20;
addpath(genpath('/home/drproduck/Documents/MATLAB'));

for mati = 1:4
    load(mat{mati});
    n = size(fea, 1);
    k = max(gnd);
    
    km_t = zeros(maxit, 1);
    cspec_acc = zeros(maxit, 10);
    cspec_t = zeros(maxit, 10);

    for r = 1:1:10
        for i = 1:maxit
            m = r * 100;
            %common representatives

            t0 = cputime;

            [lb, reps, ~, VAR] = litekmeans(fea, m, 'Distance', 'sqEuclidean', 'replicates', 1, 'maxiter',...
                10, 'Start', 'cluster', 'clustermaxrestart', 10, 'clustermaxiter', 100, 'clustersample', 0.1);

            lbcount = hist(lb, 1:m);
            cluster_sigma = sqrt(VAR ./ lbcount);
            sigma = mean(cluster_sigma);
            km_t(i) = cputime - t0;

            t0 = cputime;
            W = EuDist2(fea, reps, 0);
            W = exp(-W/(2.0*sigma^2));
        % cSPEC
            d1 = sum(W, 2);
            d1 = max(d1, 1e-100);
            d2 = sum(W, 1);
            d2 = max(d2, 1e-100);

            D1 = sparse(1:n, 1:n, d1.^(-0.5));
            D2 = sparse(1:m, 1:m, d2.^(-0.5));

            L = D1 * W * D2;

            if exist('neigv', 'var')
                [U,~,~] = mySVD(L, opts.neigv);  
            else
                [U,~,~] = mySVD(L, k);
            end

        %     U(:,1) = []; not applicable for cspec
            U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
            label = litekmeans(U, k, 'Distance', 'cosine', 'MaxIter', 100, 'Replicates',10);
            cspec_t(i,r) = cputime - t0;
            label = bestMap(gnd, label);
            cspec_acc(i,r) = sum(label == gnd) / n;
                
            fprintf('dataset = %s, landmarks = %d, iterations = %d, acc = %f\n', mat{mati}, m, i, cspec_acc(i,r));
        end
    end
    fprintf('dataset = %s, average acc = %f\n', mat{mati}, mean(cspec_acc(:,r)));
    save(strcat(mat{mati}, '_m_cspec_result'), 'cspec_acc', 'km_t', 'cspec_t')
end