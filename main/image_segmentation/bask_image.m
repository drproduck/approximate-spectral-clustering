function [U,S,V] = bask_image(arr, bi,bj, li,lj, sl, sp)
    [n,m] = size(arr);
    nl = ceil(n/li);
    ml = ceil(m/lj);
    centers = zeros(nl * ml, 2);
    count = 1;
    for i = 1:li:n
        for j = 1:lj:m
            if i + li > n
                ci = i + floor ((n - i) / 2);
            else
                ci = i + floor (li / 2);
            end
            if j + lj > m
                cj = j + floor((m - j) / 2);
            else
                cj = j + floor (lj / 2);
            end
            centers(count, 1) = ci;
            centers(count, 2) = cj;
            count = count + 1;
        end
    end
    centers
    w = zeros(n * m, size(centers, 1));
    
    for i = 1:n
        for j = 1:m
            range_i = find((centers(:,1) >= i - bi) & (centers(:,1) <= i + bi));
            range_j = find((centers(:,2) >= j - bj) & (centers(:,2) <= j + bj));
            idx = intersect(range_i, range_j);
            c = centers(idx,:);
            p1 = arr(i,j);
            c(1)
            c(2)
            p2 = arr(c(1), c(2));
            a = exp(-((i - c(1))^2 + (j - c(2))^2) / (2*sl^2) - abs(p1 - p2) / (2*sp^2));
            w(i * n + j, floor(c(1) / li) * nl + floor(c(2) / lj)) = a;
        end
    end
    [U,S,V] = bask_doc_term(w, 4, 0);
            
end