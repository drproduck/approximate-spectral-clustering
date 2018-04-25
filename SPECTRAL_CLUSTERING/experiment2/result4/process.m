clear;
close all;
clc
mat = {'letter'; 'mnist'; ...
    'usps'; 'protein'};
mark = {'--*','--p','--+','--v','--x','--s','--d','--^','--v','--<','-->','--p','--h'};
marki = 1;
% for mati = 1:4
%     figure(mati)
%     title(mat{mati})
%     a = load(strcat(mat{mati},'_m_sens_result.mat'));
%     kasp = mean(a.kasp_acc, 2) * 100;
%     lsc = mean(a.lsc_acc, 2) * 100;
%     dhillon = mean(a.acc_d, 2) * 100;
%     cspec = mean(a.acc_c, 2) * 100;
%     x3 = mean(a.acc3,2) * 100;
%     x4 = mean(a.acc4,2) * 100;
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
%     plot(100:100:1000, x4(:,1), 'Marker', mark{marki+1})
% 
%     marki = 1;
%     hold off
%     legend('kasp','lsc','dhillon','cspec','x2','y2')
%     legend('Location', 'northeastoutside')
%     xlabel('# landmarks')
%     ylabel('accuracy (%)')
%     clear a;
% end
% legend('1','2','3','4','5','6','7','8','9','10')

for mati = 1:4
    figure(mati)
    title(mat{mati},'fontsize',14)
    a = load(strcat(mat{mati},'_m_sens_result.mat'));
    %km = mean(a.km_t,2);
    km=0;
    
    kasp = mean(a.kasp_t, 2) + km;
    lsc = mean(a.lsc_t, 2) + km;
    dhillon = mean(a.tid, 2) + km;
    cspec = mean(a.tic, 2) + km;
    x3 = mean(a.ti3,2) + km;
    x4 = mean(a.ti4,2) + km;
    
    time = [x4 x3 cspec lsc dhillon kasp];
    
    hold on
    for j = 1:size(time,2)
        plot(100:100:1000, time(:,j), mark{j}, 'markersize',14, 'linewidth',2)
    end
    hold off
    legend({'LBDM2Y','LBDM2X','cSPEC', 'LSC','Dhillon','KASP'})
    %legend('Location', 'northeastoutside')
    xlabel('m (# landmarks)')
    ylabel('CPU time (s)')
    xlim([100 1000])
    set(gca, 'fontsize',14)
    grid on
    clear a;
end
% legend('1','2','3','4','5','6','7','8','9','10')


