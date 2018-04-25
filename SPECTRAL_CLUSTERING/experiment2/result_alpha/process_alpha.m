clear;
close all;
clc
mat = {'letter'; 'mnist'; ...
    'usps'; 'protein'};
mark = {'--*','--p','--+','--v','--x','--s','--d','--^','--v','--<','-->','--p','--h'};
marki = 1;
for mati = 1:4
    figure(mati)
    set(gca, 'fontsize',18)
%     title(mat{mati}, 'fontsize', 18)
    title('time step vs accuracy (higher is better)', 'fontsize', 16);
    a = load(strcat(mat{mati},'_bask_result.mat'));
    x3 = mean(a.acc3,1) * 100;
    x4 = mean(a.acc4,1) * 100;
    x5 = mean(a.acc5,1) * 100;

    hold on
    
    plot(2:2:40, x4, mark{1}, 'markersize',14','linewidth',2,'Color',[0, 0.4470, 0.7410]);
    plot(2:2:40, x3, mark{2}, 'markersize',14','linewidth',2,'Color',[0.8500, 0.3250, 0.0980]);
    plot(1:2:39, x5, mark{3}, 'markersize',14','linewidth',2,'Color',[0.9290, 0.6940, 0.1250]);

    hold off
    legend({'LBDM\alphaY','LBDM\alphaX','LBDM\alpha'})
    xlabel('\alpha (time step)')
    ylabel('accuracy (%)')
%     xlim([1 10])
    fig = gcf;
    fig.PaperPositionMode = 'auto'
    fig_pos = fig.PaperPosition;
    fig.PaperSize = [fig_pos(3) fig_pos(4)];
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


