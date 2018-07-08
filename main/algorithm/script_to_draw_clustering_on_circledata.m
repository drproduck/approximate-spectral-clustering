clear;
addpath(genpath('/home/drproduck/Documents/MATLAB/SPECTRAL_CLUSTERING/'));
load('circledata_50');

%% Spectral Clustering
% opts.sigma = 30;
% label = SC(fea, 2, 'gaussian', opts);
% label = bestMap(gnd, label);
% scatter(fea(:,1), fea(:,2), 15, label,'filled','o');
% ac = sum(label == gnd) / size(gnd, 1);
% title(strcat('Spectral Clustering,',' ', num2str(ac*100),'%'),'fontsize',14);
% 
% ax = gca;
% outerpos = ax.OuterPosition;
% ti = ax.TightInset; 
% left = outerpos(1) + ti(1);
% bottom = outerpos(2) + ti(2);
% ax_width = outerpos(3) - ti(1) - ti(3);
% ax_height = outerpos(4) - ti(2) - ti(4);
% ax.Position = [left bottom ax_width ax_height];
% 
% fig = gcf;
% fig.PaperPositionMode = 'auto'
% fig_pos = fig.PaperPosition;
% fig.PaperSize = [fig_pos(3) fig_pos(4)];
% axis off
% 
% print(fig, '/home/drproduck/Desktop/paper/poster/poster/figures/twomoons_spectralclustering', '-dpdf')

%% Kmeans
% label = litekmeans(fea, 2);
% label = bestMap(gnd, label);
% scatter(fea(:,1), fea(:,2), 15, label,'filled','o');
% ac = sum(label == gnd) / size(gnd, 1);
% title(strcat('Kmeans,',' ', num2str(ac*100),'%'),'fontsize',14);
% 
% ax = gca;
% outerpos = ax.OuterPosition;
% ti = ax.TightInset; 
% left = outerpos(1) + ti(1);
% bottom = outerpos(2) + ti(2);
% ax_width = outerpos(3) - ti(1) - ti(3);
% ax_height = outerpos(4) - ti(2) - ti(4);
% ax.Position = [left bottom ax_width ax_height];
% 
% fig = gcf;
% fig.PaperPositionMode = 'auto'
% fig_pos = fig.PaperPosition;
% fig.PaperSize = [fig_pos(3) fig_pos(4)];
% axis off
% 
% print(fig, '/home/drproduck/Desktop/paper/poster/poster/figures/twomoons_kmeans', '-dpdf')

%% Gaussian mixture model
gm = fitgmdist(fea, 2);
label = cluster(gm, fea);
scatter(fea(:,1), fea(:,2), 15, label,'filled','o');
label = bestMap(gnd, label);
ac = sum(label == gnd) / size(gnd, 1);
title(strcat('Gaussian Mixture Model,',' ', num2str(ac*100),'%'),'fontsize',14);

ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];

fig = gcf;
fig.PaperPositionMode = 'auto'
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
axis off

print(fig, '/home/drproduck/Desktop/paper/poster/poster/figures/twomoons_gmm', '-dpdf')