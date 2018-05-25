clear;
close all;
clc
mat = {'letter'; 'mnist'; ...
    'usps'; 'protein'};
mark = {'--*','--p','--+','--v','--x','--s','--d','--^','--v','--<','-->','--p','--h'};
marki = 1;

for mati = 3
    figure(mati)
    set(gca, 'fontsize',18)
%     title(mat{mati},'fontsize',18)
    title('# landmarks vs accuracy (higher is better)', 'fontsize',16);
    a = load(strcat(mat{mati},'_m_sens_result.mat'));
    b = load(strcat(mat{mati},'_m_cspec_result.mat'));
    kasp = mean(a.kasp_acc, 2) * 100;
    lsc = mean(a.lsc_acc, 2) * 100;
    dhillon = mean(a.acc_d, 2) * 100;
    cspec = mean(b.cspec_acc, 1)' * 100;
    x3 = mean(a.acc3,2) * 100;
    x4 = mean(a.acc4,2) * 100;
    
    %lbdm1
    c = load(strcat(mat{mati},'_m_sens_for_lbdm1.mat'));
    lbdm1 = mean(c.lbdm1_acc_all, 2) * 100;
    
%     acc = [x4 x3 lbdm1 cspec lsc dhillon kasp];
    % without dhillon and lbdm1
    acc = [x4 x3 cspec lsc kasp];

    
    hold on
    for j = 1:size(acc,2)
        plot(100:100:1000, acc(:,j), mark{j}, 'markersize',14, 'linewidth',2)
    end
    hold off
%     lg = legend({'LBDM2Y','LBDM2X','LBDM1','cSPEC', 'LSC','Dhillon','KASP'});
    % without dhillon and lbdm1
    lg = legend({'LBDM2Y','LBDM2X','cSPEC', 'LSC' 'KASP'});
    lg.FontSize = 14;
    %legend('Location', 'northeastoutside')
    xlabel('m (# landmarks)')
    ylabel('accuracy (%)')
    xlim([100 1000])
    grid on
    % set figure size and page size, necessary as second step if saving to pdf % format
    fig = gcf;
    fig.PaperPositionMode = 'auto'
    fig_pos = fig.PaperPosition;
    fig.PaperSize = [fig_pos(3) fig_pos(4)];
    clear a;
    clear b;
%     print(fig, '/home/drproduck/Desktop/paper/poster/poster/figures/mnist_m_accuracy','-dpdf')
end

% for mati = 2
%     figure(mati)
%     set(gca, 'fontsize',18)
% %     title(mat{mati},'fontsize',18)
% %     title('# landmarks vs time', 'fontsize', 18);
%     title('# landmarks vs run-time (lower is better)', 'fontsize',16);
%     a = load(strcat(mat{mati},'_m_sens_result.mat'));
%     b = load(strcat(mat{mati},'_m_cspec_result.mat'));
%     
%     %km = mean(a.km_t,2);
%     km=0;
%     
%     kasp = mean(a.kasp_t, 2) + km;
%     lsc = mean(a.lsc_t, 2) + km;
%     dhillon = mean(a.tid, 2) + km;
%     cspec = mean(b.cspec_t, 1)' + km;
%     x3 = mean(a.ti3,2) + km;
%     x4 = mean(a.ti4,2) + km;
%     
%     %lbdm1
%     c = load(strcat(mat{mati},'_m_sens_for_lbdm1.mat'));
%     lbdm1 = mean(c.lbdm1_t_all, 2) + km;
%     
% %     time = [x4 x3 lbdm1 cspec lsc dhillon kasp];
%     % without dhillon and lbdm1
%     time = [x4 x3 cspec lsc kasp];
%     
%     hold on
%     for j = 1:size(time,2)
%         plot(100:100:1000, time(:,j), mark{j}, 'markersize',14, 'linewidth',2)
%     end
%     hold off
% %     legend({'LBDM2Y','LBDM2X','LBDM1','cSPEC', 'LSC','Dhillon','KASP'})
%     % without dhillon and lbdm1
%     lg = legend({'LBDM2Y','LBDM2X','cSPEC', 'LSC', 'KASP'});
%     lg.FontSize = 14;
%     %legend('Location', 'northeastoutside')
%     xlabel('m (# landmarks)')
%     ylabel('CPU time (s)')
%     xlim([100 1000])
%     fig = gcf;
%     fig.PaperPositionMode = 'auto';
%     fig_pos = fig.PaperPosition;
%     fig.PaperSize = [fig_pos(3) fig_pos(4)];
%     grid on
%     clear a;
%     clear b;
% end



