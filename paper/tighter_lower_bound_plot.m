clear;
k1 = 8:1:5000;
log_2overk = log(2) - log(k1);
b1 = (2+5.*log_2overk+sqrt(9.*(log_2overk.^2)+4))/(2.*log_2overk);
log_a1=b1.*log(2)-log(1-b1)-log(4-b1);
log_obj=log_a1+log(k1).*(1-b1);
plot(k1,exp(log_obj)./log(k1));
xlabel('$k$','Fontsize',20, 'Interpreter','latex');
ylabel('$\frac{\bar{a}_{k}k^{1-\bar{b}_{k}}}{\log k}$','Fontsize',20, 'Interpreter','latex');
plt=gca;
exportgraphics(plt,'tighter_lower_bound_plot.eps','Colorspace','rgb','Resolution',600);
