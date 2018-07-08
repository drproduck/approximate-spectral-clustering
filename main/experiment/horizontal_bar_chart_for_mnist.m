c = categorical({'LBDM2Y','LBDM2X','LBDM1','cSPEC','LSC','Dhillon','KASP'});
y = [73.29;72.37;72.43;54.50;70.28;72.15;57.99];

figure;
bar(c,y,'grouped'); % groups by row

set(gca, 'fontsize',16)
title('accuracy', 'fontsize',16);
% xlim([50 100])
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];