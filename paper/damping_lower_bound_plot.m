k1 = 2:1:1000;
b1 = 0:1e-3:1;
b1 = b1(2:length(b1)-1);
[k, b] = meshgrid(k1, b1);
Z = ((2./k).^b)./((1-b).*(4-b));
surf(k,b,Z);
zlim([-3,3]);