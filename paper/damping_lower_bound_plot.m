k1 = 1:1:1000;
b1 = 0:1e-3:1;
b1 = b1(2:length(b1)-1);
[k, b] = meshgrid(k1, b1);
Z = ((2./k).^b)./((1-b).*(4-b));
meshc(k,b,Z);
zlim(-5,5);