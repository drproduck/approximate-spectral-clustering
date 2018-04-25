load('TDT2_new')
mark = {'--*','--p','--+','--v','--x','--s','--d','--^','--v','--<','-->','--p','--h'};

set(gca, 'fontsize',18)
title('TDT2','fontsize',18);
x = mean(acc3);
y = mean(acc4);
xy = mean(acc5);
dhillon = mean(acc_d);

acc = [x y xy dhillon];
hold on
plot(2:2:20, x, mark{2}, 'markersize',14,'linewidth',2,'Color',[0.8500, 0.3250, 0.0980])
plot(1:2:19, xy, mark{3}, 'markersize',14,'linewidth',2,'Color',[0.9290, 0.6940, 0.1250])
plot(0, dhillon, mark{6}, 'markersize',14,'linewidth',2,'Color',[0.3010, 0.7450, 0.9330])
% plot(2:2:20, y, mark{2}, 'markersize',14,'linewidth',2)
% plot(1, cspec, mark{5}, 'markersize',14,'linewidth',2)
hold off
legend('LBDM\alphaX', 'LBDM\alpha', 'Dhillon')
xlabel('\alpha (time step)')
ylabel('accuracy (%)')

grid on

% marki = 1;
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
%     acc = [x4 x3 cspec lsc dhillon kasp];
%     hold on
%     
%     for j = 1:size(acc,2)
%         plot(100:100:1000, acc(:,j), mark{j}, 'markersize',14, 'linewidth',2)
%     end
%     hold off
%     legend({'LBDM2Y','LBDM2X','cSPEC', 'LSC','Dhillon','KASP'})
%     %legend('Location', 'northeastoutside')
%     xlabel('m (# landmarks)')
%     ylabel('accuracy (%)')
%     xlim([100 1000])
%     set(gca, 'fontsize',14)
%     grid on
%     clear a;
% end
