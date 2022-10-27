k1 = 2:1:2000;
b1 = 0:1e-3:1;
b1 = b1(2:length(b1)-1); % since b can't take 0 or 1
[k, b] = meshgrid(k1, b1);
Z = ((2./k).^b)./((1-b).*(4-b));
plt = surfc(k,b,log(Z),log(Z));
xlabel('$k$','Fontsize',12, 'Interpreter','latex');
ylabel('$b$','Fontsize',12, 'Interpreter','latex');
zlabel('$\log(\bar{a}k^{-b})$','Fontsize',12, 'Interpreter','latex');
hold on;
plot3(k(log(Z)==min(log(Z))),b(log(Z)==min(log(Z))),min(log(Z)),"r");
hold off;
set(plt,"LineStyle","none");
colorbar;
view(-45,45);