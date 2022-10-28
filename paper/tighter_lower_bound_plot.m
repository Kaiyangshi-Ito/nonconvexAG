clear;
k1 = 8:1:1e9;
log_2overk = log(2) - log(k1);
b1 = (2+5.*log_2overk+sqrt(9.*(log_2overk.^2)+4))/(2.*log_2overk);
a1=2.^b1/((1-b1).*(4-b1));
obj=a1.*k1.^(1-b1);
plt=plot(log(k1),log(obj));
xlabel('$\log(k)$','Fontsize',20, 'Interpreter','latex');
ylabel('$\log\left(\bar{a}k^{1-\bar{b}}\right)$','Fontsize',20, 'Interpreter','latex');
plt=gca;
exportgraphics(plt,'tighter_lower_bound_plot.eps','Colorspace','rgb','Resolution',600);
fd=(log(obj(2:length(obj)))-log(obj(1:length(obj)-1)))./(log(k1(2:length(k1)))-log(k1(1:length(k1)-1)));
long(max(fd))
long(min(fd))