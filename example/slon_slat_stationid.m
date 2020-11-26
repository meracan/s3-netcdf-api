url = 'https://api.meracan.ca/';
sdata = 'slon,slat,stationid';

data = webread(url, 'variable', sdata);

sx = cell2mat(struct2cell(data.longitude_degrees_east));
sy = cell2mat(struct2cell(data.latitude_degrees_north));
sc = cell2mat(struct2cell(data.StationId));

scatter(sx,sy,[],sc);
title('spectra node locations', 'FontSize', 20);
xlabel('longitude', 'FontSize', 20);
ylabel('latitude', 'FontSize', 20);
grid on;