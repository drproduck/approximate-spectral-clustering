clear;
close all;
mat = {'letter'; 'mnist'; ...
    'usps'; 'protein'};
mark = {'--*','--p','--+','--v','--x','--s','--d','--^','--v','--<','-->','--p','--h'};
marki = 1;
addpath(genpath('/home/drproduck/Documents/MATLAB'));
for mati = 1:4
    figure(mati)
    title(mat{mati})
    a = load(strcat(mat{mati},'_s_sens_result.mat'));
    b = load(strcat(mat{mati},'_cspec_result.mat'));
    x4 = mean(a.acc4,2)*100;
    x3 = mean(a.acc3,2)*100;
    cspec = mean(b.cspec_acc,1)'*100;
    lsc = mean(a.lsc_acc, 2)*100;
    dhillon = mean(a.acc_d, 2)*100;
    kasp = mean(a.kasp_acc, 2)*100;
       
    hold on   
    plot(2:1:10, x4(2:10), mark{1}, 'markersize',14, 'linewidth',2)
    plot(2:1:10, x3(2:10), mark{2}, 'markersize',14, 'linewidth',2)
    plot(0, mean(cspec), mark{3},'markersize',14, 'linewidth',2)
    plot(2:1:10, lsc(2:10), mark{4}, 'markersize',14, 'linewidth',2)
    plot(2:1:10, dhillon(2:10), mark{5}, 'markersize',14, 'linewidth',2)
    plot(1, mean(kasp), mark{6},'markersize',14, 'linewidth',2);
    
    hold off
    legend({'LBDM2Y','LBDM2X','cSPEC','LSC','Dhillon','KASP'})
    xlabel('s (#nearest landmarks)')
    ylabel('accuracy (%)')
    xlim([0 10])
    set(gca, 'fontsize',14)
    grid on
    clear a;
end
% legend('1','2','3','4','5','6','7','8','9','10')

% for mati = 1:4
%     figure(mati)
%     title(mat{mati})
%     a = load(strcat(mat{mati},'_s_sens_result.mat'));
%     b = load(strcat(mat{mati},'_cspec_result.mat'));
%     x4 = mean(a.ti4,2);
%     x3 = mean(a.ti3,2);
%     cspec = mean(b.cspec_t, 1)';
%     lsc = mean(a.lsc_t, 2);
%     dhillon = mean(a.tid, 2);
%     kasp = mean(a.kasp_t, 2);
%        
%     hold on
%     plot(2:1:10, x4(2:10), mark{1}, 'markersize',14, 'linewidth',2)
%     plot(2:1:10, x3(2:10), mark{2}, 'markersize',14, 'linewidth',2)
%     plot(0, mean(cspec), mark{3},'markersize',14, 'linewidth',2)
%     plot(2:1:10, lsc(2:10), mark{4}, 'markersize',14, 'linewidth',2)
%     plot(2:1:10, dhillon(2:10), mark{5}, 'markersize',14, 'linewidth',2)
%     plot(1, mean(kasp), mark{6},'markersize',14, 'linewidth',2);
%     hold off
%     legend({'LBDM2Y','LBDM2X','cSPEC','LSC','Dhillon','KASP'})
%     xlabel('s (#nearest landmarks)')
%     ylabel('time (s)')
%     xlim([0 10])
%     set(gca, 'fontsize',14)
%     grid on
%     clear a;
% end



