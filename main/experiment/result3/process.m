clear;
mat = {'letter'; 'mnist'; 'musk_1'; ...
    'pend'; 'shuttle'; 'usps'; 'protein'; 'connect4_binary_1';'news100'};
mark = {'+','o','*','.','x','s','d','^','v','<','>','p','h'};
marki = 1;
% for mati = 1:9
%     figure(mati)
%     title(mat{mati})
%     a = load(strcat(mat{mati},'_m_sens_result.mat'));
%     kasp = mean(a.kasp_acc, 2) * 100;
%     lsc = mean(a.lsc_acc, 2) * 100;
%     dhillon = mean(a.acc_d, 2) * 100;
%     cspec = mean(a.acc_c, 2) * 100;
%     x3 = reshape(mean(a.acc3,2), 10, 10) * 100;
%     x4 = reshape(mean(a.acc4,2), 10, 10) * 100;
%     x5 = reshape(mean(a.acc5,2), 10, 10) * 100;
%     
%     hold on
%     plot(100:100:1000, kasp, 'Marker', mark{marki+1})
%     marki = marki + 1;
%     plot(100:100:1000, lsc, 'Marker', mark{marki+1})
%     marki = marki + 1;
%     plot(100:100:1000, dhillon, 'Marker', mark{marki+1})
%     marki = marki + 1;
%     plot(100:100:1000, cspec, 'Marker', mark{marki+1})
%     marki = marki + 1;
%     plot(100:100:1000, x3(:,1), 'Marker', mark{marki+1})
%     marki = marki + 1;
%     plot(100:100:1000, x3(:,2), 'Marker', mark{marki+1})
%     marki = marki + 1;
%     plot(100:100:1000, x4(:,1), 'Marker', mark{marki+1})
%     marki = marki + 1;
%     plot(100:100:1000, x4(:,2), 'Marker', mark{marki+1})
%     marki = marki + 1;
%     plot(100:100:1000, x5(:,1), 'Marker', mark{marki+1})
%     marki = marki + 1;
%     plot(100:100:1000, x5(:,2), 'Marker', mark{marki+1})
%     marki = 1;
%     hold off
%     legend('kasp','lsc','dhillon','cspec','x2','x4','y2','y4','1','3')
%     legend('Location', 'northeastoutside')
%     xlabel('# landmarks')
%     ylabel('accuracy (%)')
%     clear a;
% end
% legend('1','2','3','4','5','6','7','8','9','10')

for mati = 9:9
    figure(mati)
    title(mat{mati})
    a = load(strcat(mat{mati},'_m_sens_result.mat'));
    km = mean(a.km_t,2);
    
    kasp = mean(a.kasp_t, 2) + km;
    lsc = mean(a.lsc_t, 2) + km;
    dhillon = mean(a.tid, 2) + km;
    cspec = mean(a.tic, 2) + km;
    x3 = reshape(mean(a.ti3,2), 10, 10) + km;
    x4 = reshape(mean(a.ti4,2), 10, 10) + km;
    x5 = reshape(mean(a.ti5,2), 10, 10) + km;
    
    hold on
    plot(100:100:1000, kasp, 'Marker', mark{marki+1})
    marki = marki + 1;
    plot(100:100:1000, lsc, 'Marker', mark{marki+1})
    marki = marki + 1;
    plot(100:100:1000, dhillon, 'Marker', mark{marki+1})
    marki = marki + 1;
    plot(100:100:1000, cspec, 'Marker', mark{marki+1})
    marki = marki + 1;
    plot(100:100:1000, x3(:,1), 'Marker', mark{marki+1})
    marki = marki + 1;
    plot(100:100:1000, x3(:,2), 'Marker', mark{marki+1})
    marki = marki + 1;
    plot(100:100:1000, x4(:,1), 'Marker', mark{marki+1})
    marki = marki + 1;
    plot(100:100:1000, x4(:,2), 'Marker', mark{marki+1})
    marki = marki + 1;
    plot(100:100:1000, x5(:,1), 'Marker', mark{marki+1})
    marki = marki + 1;
    plot(100:100:1000, x5(:,2), 'Marker', mark{marki+1})
    marki = 1;
    hold off
    legend('kasp', 'lsc','dhillon','cspec','x2','x4','y2','y4','1','3')
    legend('Location', 'northeastoutside')
    xlabel('# landmarks')
    ylabel('time (s)')
    clear a;
end
% legend('1','2','3','4','5','6','7','8','9','10')


