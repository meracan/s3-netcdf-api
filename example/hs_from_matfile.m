node = 1;
ntimes = 745;
HS = load('~/Desktop/results/2004/01/results/HS.mat');
fn = fieldnames(HS);

fns = string(fn);
fns = extractAfter(fns, 'Hsig_');
fns = strrep(fns, "_", " ");
hs = zeros(1, ntimes);
for i = 1:ntimes
 hs(i) = HS.(fn{i})(node);
end
plot(hs)
title('Significant wave height for Jan 2004', 'FontSize', 20);
xtickangle(15);
xticklabels(fns);
xlabel('timestep (h)', 'FontSize', 20);
ylabel('wave height (HS), in metres', 'FontSize', 20);
grid on;