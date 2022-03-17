
y = [90.2118364  87.31436548 91.28063255];
err = [3.19328953 3.0636404  5.51530578];




b = bar(y,0.5)
set(gca,'XTickLabel',{'Direct-S','Direct-L','GIF'});
set(gca,'box','off');
set(gca,'tickdir','out');
set(gca,'yminortick','off');
set(0, 'DefaultAxesFontName', 'Arial');
t = title('Path efficiency (Euclidean distance / Path length)')
t.FontSize = 16;

ylabel('Efficiency (%)')
axis([0.5,3.5,80,100])
b.FaceColor = 'flat';
b.CData(1,:) = [185 42 42]/255;
b.CData(2,:)= [0 0.6 0];
b.CData(3,:) = [70 130 190]/255;

hold on
errorbar(y,err,'k', 'Linestyle', 'None','HandleVisibility','off')

hold off
