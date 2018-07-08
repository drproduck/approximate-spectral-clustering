clear;
close all;
clc
mat = {'letter'; 'mnist'; ...
    'usps'; 'protein'};
mark = {'--*','--p','--+','--v','--x','--s','--d','--^','--v','--<','-->','--p','--h'};
marki = 1;
for mati = 1:4
    figure(mati)
    title(mat{mati})
    a = load(strcat(mat{mati},'_bask_result.mat'));
    x3 = mean(a.acc3,1) * 100;
    x4 = mean(a.acc4,1) * 100;
    x5 = mean(a.acc5,1) * 100;

    hold on
    
    plot(2:2:40, x4, mark{marki}, 'markersize',14','linewidth',2);
    marki = marki + 1;
    plot(2:2:40, x3, mark{marki}, 'markersize',14','linewidth',2);
    marki = marki + 1;
    plot(1:2:39, x5, mark{marki}, 'markersize',14','linewidth',2);
    marki = 1;
    
    hold off
    legend({'LBDM\alphaY','LBDM\alphaX','LBDM\alpha'})
    xlabel('alpha (time step)')
    ylabel('accuracy (%)')
%     xlim([1 10])
    set(gca, 'fontsize',14)
    grid on
    clear a;
end
% legend('1','2','3','4','5','6','7','8','9','10')

% for mati = 1:4
%     figure(mati)
%     title(mat{mati},'fontsize',14)
%     a = load(strcat(mat{mati},'_s_sens_result.mat'));
%     %km = mean(a.km_t,2);
%     km=0;
%     
%     kasp = mean(a.kasp_t, 2) + km;
%     lsc = mean(a.lsc_t, 2) + km;
%     dhillon = mean(a.tid, 2) + km;
%     cspec = mean(a.tic, 2) + km;
%     x3 = mean(a.ti3,2) + km;
%     x4 = mean(a.ti4,2) + km;
%     
%     time = [x4 x3 cspec lsc dhillon kasp];
%     
%     hold on
%     for j = 1:size(time,2)
%         plot(1:1:10, time(:,j), mark{j}, 'markersize',14, 'linewidth',2)
%     end
%     hold off
%     legend({'LBDM2Y','LBDM2X','cSPEC', 'LSC','Dhillon','KASP'})
%     %legend('Location', 'northeastoutside')
%     xlabel('s (# nearest neighbors')
%     ylabel('CPU time (s)')
%     xlim([1 10])
%     set(gca, 'fontsize',14)
%     grid on
%     clear a;
% end
% legend('1','2','3','4','5','6','7','8','9','10')


