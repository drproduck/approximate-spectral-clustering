clear;
close all;
mat = {'letter'; 'mnist'; ...
    'usps'; 'protein'};
mark = {'--*','--p','--+','--v','--x','--s','--d','--^','--v','--<','-->','--p','--h'};
marki = 1;
addpath(genpath('/home/drproduck/Documents/MATLAB'));

% for mati = 2
%     figure(mati)
%     set(gca, 'fontsize',18)
% %     title(mat{mati}, 'fontsize', 18)
%     title('# nearest landmarks vs accuracy', 'fontsize',16);
%     a = load(strcat(mat{mati},'_s_sens_result.mat'));
%     b = load(strcat(mat{mati},'_cspec_result.mat'));
%     x4 = mean(a.acc4,2)*100;
%     x3 = mean(a.acc3,2)*100;
%     cspec = repmat(mean(b.cspec_acc,1)*100, 10, 1);
%     lsc = mean(a.lsc_acc, 2)*100;
%     dhillon = mean(a.acc_d, 2)*100;
%     kasp = repmat(mean(mean(a.kasp_acc))*100, 10, 1);
%     
%     %load matrix for lbdm1
%     c = load(strcat(mat{mati},'_s_sens_for_lbdm1.mat'));
%     lbdm1 = mean(c.lbdm1_acc_all,2)*100;
%        
%     
%     acc = [x4 x3 cspec lsc kasp];
%     
%     hold on
%     for j = 1:size(acc,2)
%         plot(2:1:10, acc(2:10,j), mark{j}, 'markersize',14, 'linewidth',2)
%     end
%     hold off
% %     lg = legend({'LBDM2Y','LBDM2X','LBDM1','cSPEC','LSC','Dhillon','KASP'});
%     lg = legend({'LBDM2Y','LBDM2X','cSPEC','LSC','KASP'});
%     lg.FontSize = 14;
%     xlabel('s (#nearest landmarks)')
%     ylabel('accuracy (%)')
%     xlim([2 10])
%     set(gca, 'xtick', 0:2:10)
%     grid on
%     fig = gcf;
%     fig.PaperPositionMode = 'auto';
%     fig_pos = fig.PaperPosition;
%     fig.PaperSize = [fig_pos(3) fig_pos(4)];
%     clear a;
%     clear b;
% end
% 
for mati = 2
    figure(mati)
    set(gca, 'fontsize',18)
%     title(mat{mati}, 'fontsize', 18)
    title('# nearest landmarks vs run-time', 'fontsize',16);
    a = load(strcat(mat{mati},'_s_sens_result.mat'));
    b = load(strcat(mat{mati},'_cspec_result.mat'));
    x4 = mean(a.ti4,2);
    x3 = mean(a.ti3,2);
    cspec = repmat(mean(b.cspec_t, 1), 10, 1);
    lsc = mean(a.lsc_t, 2);
    dhillon = mean(a.tid, 2);
    kasp = repmat(mean(mean(a.kasp_t)), 10, 1);
    
    %load matrix for lbdm1
    c = load(strcat(mat{mati},'_s_sens_for_lbdm1.mat'));
    lbdm1 = mean(c.lbdm1_t_all,2);
       
    time = [x4 x3 cspec lsc kasp];
    
    hold on
    for j = 1:size(time,2)
        plot(2:1:10, time(2:10,j), mark{j}, 'markersize',14, 'linewidth',2)
    end
    hold off
%     lg = legend({'LBDM2Y','LBDM2X','LBDM1','cSPEC','LSC','Dhillon','KASP'});
    lg = legend({'LBDM2Y','LBDM2X','cSPEC', 'LSC', 'KASP'});
    lg.FontSize = 14;
    xlabel('s (#nearest landmarks)')
    ylabel('CPU time (s)')
    xlim([2 10])
    set(gca, 'xtick', 0:2:10)
    grid on
    fig = gcf;
    fig.PaperPositionMode = 'auto';
    fig_pos = fig.PaperPosition;
    fig.PaperSize = [fig_pos(3) fig_pos(4)];
    clear a;
    clear b;
end



