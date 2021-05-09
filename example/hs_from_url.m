url = 'https://api.meracan.ca/?';
var = 'hs';
x = -136.7264224;
y = 57.39504017;
s = '2004-01-01';
e = '2004-02-01';

data = webread(url, 'variable',var, 'x',x, 'y',y, 'start',s, 'end',e);

times = cell2mat(struct2cell(data.Datetime));
hs = cell2mat(struct2cell(data.hs_m));

plot(hs);
title('Significant wave height', 'FontSize', 20);
xtickangle(15);
xticklabels(times);
xlabel('timestep (h)', 'FontSize', 20);
ylabel('wave height (HS), in metres', 'FontSize', 20);
grid on;



