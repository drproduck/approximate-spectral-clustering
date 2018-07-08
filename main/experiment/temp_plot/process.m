clear;
close all;
clc
addpath(genpath('/home/drproduck/Documents/MATLAB/'));
mat = {'usps','pend','letter','protein','shuttle','mnist'};
mark = {'--*','--p','--+','--v','--x','--s','--d','--^','--v','--<','-->','--p','--h'};
lbdm2y = zeros(6,1);
lbdm2x = zeros(6,1);
cspec = zeros(6,1);
lsc = zeros(6,1);
dhillon = zeros(6,1);
kasp = zeros(6,1);
lbdm1 = zeros(6,1);
lbdm3 = zeros(6,1);

% for mati = 1:6
% 
%     a = load(strcat(mat{mati},'_bask_result.mat'));
%     b = load(strcat(mat{mati},'_cspec_result.mat'));
%      
%     lbdm2y(mati) = mean(a.acc4(:,1)) * 100;
%     lbdm2x(mati) = mean(a.acc3(:,1)) * 100;
%     cspec(mati) = mean(b.cspec_acc) * 100;
%     lsc(mati) = mean(a.lsc_acc) * 100;
%     dhillon(mati) = mean(a.acc6) * 100;
%     kasp(mati) = mean(a.kasp_acc) * 100;
%     lbdm1(mati) = mean(a.acc5(:,1)) * 100;
%     lbdm3(mati) = mean(a.acc5(:,2)) * 100;
%     
%     clear a;
%     clear b;
% 
% end
% 
% acc = [lbdm2y lbdm2x cspec lsc dhillon kasp lbdm1 lbdm3];
% hold on
% for i = 1:size(acc, 2)
%     plot(1:6, acc(:,i), mark{1}, 'markersize',14','linewidth',2);
% end
% hold off
% legend({'lbdm^{(2y)}' 'lbdm^{(2x)}' 'cspec' 'lsc' 'dhillon' 'kasp' 'lbdm^{(1)}' 'lbdm^{(3)}'})
% xlabel('dataset')
% xticklabels(mat);
% ylabel('accuracy (%)')
% %     xlim([1 10])
% set(gca, 'fontsize',14)
% grid on
    


load('temp');

lbdm2y = a(:,8);
lbdm2x = a(:,7);
cspec = a(:,3);
lsc = a(:,2);
dhillon = a(:,4);
kasp = a(:,1);
lbdm1 = a(:,5);
lbdm3 = a(:,6);
    
acc = [lbdm2y lbdm2x cspec lsc dhillon kasp lbdm1 lbdm3];
hold on
for i = 1:size(acc, 2)
    plot(1:6, acc(:,i), mark{i}, 'markersize',14','linewidth',2);
end
hold off
legend({'LBDM2Y' 'LBDM2X' 'cSPEC' 'LSC' 'Dhillon' 'KASP' 'LBDM1' 'LBDM3'})
xlabel('dataset')
xticklabels(mat);
ylabel('time (s)')
%     xlim([1 10])
set(gca, 'fontsize',14)
grid on


