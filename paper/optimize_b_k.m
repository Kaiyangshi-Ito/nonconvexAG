clear;
k1 = 1:1:5000;
b1 = 0:1e-4:1;
b1 = b1(2:length(b1)-1); % since b can't take 0 or 1
[k, b] = meshgrid(k1, b1);
log_Z = (b.*log(2./k))-log(1-b)-log(4-b);
plt = surfc(k,b,log_Z,log_Z);
xlabel('$k$','Fontsize',12, 'Interpreter','latex');
ylabel('$b$','Fontsize',12, 'Interpreter','latex');
zlabel('$\log(\bar{a}_{k}k^{-b})$','Fontsize',12, 'Interpreter','latex');
hold on;
plot3(k(log_Z==min(log_Z)),b(log_Z==min(log_Z)),min(log_Z),"r","LineWidth",1.5);
hold off;
set(plt,"LineStyle","none");
colorbar;
view(-45,45);
plt=gca;
exportgraphics(plt,'optimize_b_k.eps','Colorspace','rgb','Resolution',600);

