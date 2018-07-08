% give labels of r-nearest reps, pick label with the most occurence. 
        can_reps_label = find(W(i,:) ~= 0);
        for j = 1:size(can_reps_label,2)
            can_reps_label(j) = reps_label(can_reps_label(j));
        end
        [~, m] = max(hist(can_reps_label, 1:k));
        l4(i,T) = m;