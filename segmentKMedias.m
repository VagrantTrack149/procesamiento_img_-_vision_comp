close all;
clc;
clear all;

%im1 = imread('./images/windex.bmp');
% im1= imread('./images/Pimientos.jpg');
im1= imread('./images/tablero.jpg');

h=fspecial("gaussian",[7,7],3);
im2=imfilter(im1,h,"replicate");
%figure(1)
%imshow(im2)
im = rgb2lab(im2);
%im = uint8(im);
im = single(im);

npoints = 5;

[L,centers]=imsegkmeans(im,npoints);
[m,n,ch] = size(im);
pixels = reshape(im, m*n, 3);

for k=1:npoints
    imRes = zeros(m,n,ch);
    imR=zeros(m,n);
    imBin = zeros(m,n);

    Lab_ref = [centers(k,1);centers(k,2);centers(k,3)];

    imR(:,:,1) =(im(:,:,1)-Lab_ref(1)).^2;
    imR(:,:,2) =(im(:,:,2)-Lab_ref(2)).^2;
    imR(:,:,3) =(im(:,:,3)-Lab_ref(3)).^2;
    
    imR =sqrt(imR(:,:,1) + imR(:,:,2) + imR(:,:,3));
            
    imR = imR/max(max(imR));
    th = 0.22;
    imBin(imR<th)=1;

    imRes(:,:,1)= single(im1(:,:,1)).*single(imBin(:,:));
    imRes(:,:,2)= single(im1(:,:,2)).*single(imBin(:,:));
    imRes(:,:,3)= single(im1(:,:,3)).*single(imBin(:,:));

    figure(2)
    imshowpair(im1,uint8(imRes),'montage');
    title('Segmentacion')
    pause()
end

%% segmentando KNN
labels = [1;2;3;4;5];
%entrenamiento por Knn
knn = fitcknn(centers, labels, 'NumNeighbors',1, 'Standardize',true);
predicted_labels = predict(knn,pixels);

segmented_img = reshape(predicted_labels, m, n);

segmentad_img = label2rgb(segmented_img);

figure(3)
imshow(segmentad_img)
title('Segmentacion KNN')
%colormap(jet);
%colorbar;



