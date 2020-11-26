url = 'https://api.meracan.ca/';
vars = 'u10,v10';
itime = '7590';
inode = ':50000';

options = weboptions;
options.Timeout = Inf;
data = webread(url, 'variable', vars, 'itime', itime, options);

x = cell2mat(struct2cell(data.longitude_degrees_east));
y = cell2mat(struct2cell(data.latitude_degrees_north));
u = cell2mat(struct2cell(data.u10_m_s));
v = cell2mat(struct2cell(data.v10_m_s));

quiver(x,y,u,v);

title('wind velocity at time 7590', 'FontSize', 20);
xlabel('longitude', 'FontSize', 20);
ylabel('latitude', 'FontSize', 20);
grid on;