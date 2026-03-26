%close all;
clear all;
clc;

A = imread('./FiltradoFourier/windex.bmp');
Acie50 = rgb2lab(A, 'WhitePoint', 'd50');
Acie65 = rgb2lab(A, 'WhitePoint', 'd65');
%figure(1)
%imshow(A);
%title('Imagen original');

figure(6)
imshowpair(Acie50(:,:,1), Acie65(:,:,1), 'montage');
title('Imagen L cielab D50 vs D65');

figure(7)
imshowpair(Acie50(:,:,2), Acie65(:,:,2), 'montage');
title('Imagen A cielab D50 vs D65');

figure(8)
imshowpair(Acie50(:,:,3), Acie65(:,:,3), 'montage');
title('Imagen B cielab D50 vs D65');



%% Conversion RGB to CIELAB
close all;
clear all;
clc;

%A = imread('./FiltradoFourier/windex.bmp');
A = imread('./images/windex.bmp');

figure(1)
imshow(A);
title('Imagen original');

[u,v,ch] =  size(A);
% numero de puntos para interpolar el color
np = 4;
[x, y] = ginput(np);
x = round(x);
y = round(y);

ng = 255;
An = A/ng;

imlab = rgb2lab(A, 'WhitePoint', 'd50');
%imlab= rgb2hsv(A);
figure(2)
imshow(imlab(:,:,1), [min(min(imlab(:,:,1))) max(max(imlab(:,:,1)))]);
title('Componente L');

figure(3)
imshow(imlab(:,:,2), [min(min(imlab(:,:,2))) max(max(imlab(:,:,2)))]);
title('Componente a');


figure(4)
imshow(imlab(:,:,3), [min(min(imlab(:,:,3))) max(max(imlab(:,:,3)))]);
title('Componente b');

auxlab = zeros(3, v*u);
ax = imlab(:,:,1);
auxlab(1,:) = ax(:);
ax = imlab(:,:,2);
auxlab(2,:) = ax(:);
ax = imlab(:,:,3);
auxlab(3,:) = ax(:);

% Segmentacion de uno o varios colores
p_c = zeros(2,np);
nf=2;
% Segmentacion
for k=1:np
    
    % Valor del color a segmentar
    Lab_ref = [imlab(y(k),x(k),1); imlab(y(k),x(k),2); imlab(y(k),x(k),3)];
    th = 0.17;

    % distancia con respecto a toda la imagen en cada
    % componente
    mSeg = ((auxlab(1, :)-Lab_ref(1)).^2+...
            (auxlab(2, :)-Lab_ref(2)).^2+...
            (auxlab(3, :)-Lab_ref(3)).^2).^(1/2);

    imgProb = zeros(u, v);

    imgProb(:) = (mSeg)/max(mSeg);

    imgMasc = zeros(u, v);
    imgMasc(imgProb < th) = 1;
 
    figure(nf);
    imshow(imgProb)
    title('Imagen de probabilidad');
    nf=nf+k;
    figure(nf)
    imshow(imgMasc);
    title('Imagen de mascara');
    
   % imwrite(imgMasc, 'windexMask.jpg');
    
    % Filtrado de imagen

    %se = strel('line',11,90);
    se = strel('line',3,90);
    imgMasc_f = imerode(imgMasc,se);

    nf=nf+k;
    figure(nf);
    imshow(imgMasc_f);
title('Imagen de mascara erosionada');
    
% Punto de la region
   % rg = bwconncomp(imgMasc_f);
   % rg_d = regionprops(rg, 'basic');

   % p_c(:,k) = rg_d.Centroid;
   imRes = zeros(u, v, 3);
ind = imgMasc_f == 1;

imRes(:,:,1)=single(imgMasc_f).*single(A(:,:,1));
imRes(:,:,2)=single(imgMasc_f).*single(A(:,:,2));
imRes(:,:,3)=single(imgMasc_f).*single(A(:,:,3));

nf=nf+k;
figure(nf);
    imshow(uint8(imRes));
title('Imagen Resultante');
pause()

end

%%

% pasar al espacio XYZ

mT = [0.412453 0.357580 0.180423; 
      0.212671 0.715160 0.072169;
      0.019334 0.119193 0.950227];
  
br = [0.9504; 1.0; 1.088754];
[u,v,ch] =  size(A);

imAux = zeros(3, u*v);

imren = An(:,:,1);
imAux(1,:) = imren(:);

imren = An(:,:,2);
imAux(2,:) = imren(:);

imren = An(:,:,3);
imAux(3,:) = imren(:);

imXYZ = mT*imAux;

imXYZ(1,:)= imXYZ(1,:)/br(1);
imXYZ(2,:)= imXYZ(2,:)/br(2);
imXYZ(3,:)= imXYZ(3,:)/br(3);

val = (6/29)^3;
resf = zeros(3, u*v);
resf(1, imXYZ(1,:) > val) =  imXYZ(1, imXYZ(1,:) > val).^(1/3);
resf(1, imXYZ(1,:) <= val) =  ((29/6)^2).*imXYZ(1, imXYZ(1,:)<= val)./3+(4/29);

resf(2, imXYZ(2,:) > val) =  imXYZ(2, imXYZ(2,:) > val).^(1/3);
resf(2, imXYZ(2,:) <= val) =  ((29/6)^2).*imXYZ(2, imXYZ(2,:)<= val)./3+(4/29);

resf(3, imXYZ(3,:) > val) =  imXYZ(3, imXYZ(3,:) > val).^(1/3);
resf(3, imXYZ(3,:) <= val) =  ((29/6)^2).*imXYZ(3, imXYZ(3,:)<= val)./3+(4/29);

auxlab = zeros(3, v*u);

auxlab(1,:) = 116.*resf(2,:)-16;
auxlab(2,:) = 500.*(resf(1,:)-resf(2,:));
auxlab(3,:) = 200.*(resf(2,:)-resf(3,:));

imlab = zeros(u,v,3);

imlab(:,:,1) = reshape(auxlab(1,:),u,v);
imlab(:,:,2) = reshape(auxlab(2,:),u,v);
imlab(:,:,3) = reshape(auxlab(3,:),u,v);

