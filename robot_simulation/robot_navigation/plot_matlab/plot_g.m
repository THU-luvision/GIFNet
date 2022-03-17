
y = [34.04564933 33.54221658 29.08233835];
err = [ 9.64896893 10.40800297 13.53590652];





b = bar(y,0.5)
set(gca,'XTickLabel',{'Direct-S','Direct-L','GIF'});
set(gca,'box','off');
set(gca,'tickdir','out');
set(gca,'yminortick','off');
set(0, 'DefaultAxesFontName', 'Arial');
t = title('Attention irregularity (Rototation * Angle)')
t.FontSize = 16;

ylabel('Irregularity increment')
axis([0.5,3.5,0,55])
b.FaceColor = 'flat';
b.CData(1,:) = [185 42 42]/255;
b.CData(2,:)= [0 0.6 0];
b.CData(3,:) = [70 130 190]/255;

hold on


errorbar(y,err,'k', 'Linestyle', 'None','HandleVisibility','off')

hold off
