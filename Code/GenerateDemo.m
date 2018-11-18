clear all
close all
clc

% delete(strcat('./Mat/*.mat'));
% delete(strcat('./Edge/R-3/*.mat'));

Labels = dir('./Label/*.png');

sum2 = 0;
sum3 = 0;
sum4 = 0;

for index = 1:length(Labels)
    
    image = imread(strcat('./Image/', Labels(index).name));
    [height, width] = size(image);
    
    label = imread(strcat('./Label/', Labels(index).name));
    label(label>0) = 1;
    
    contour_map = edge(label, 'canny');
    
    name = strcat('./Mat/', Labels(index).name(1:end-4), '.mat');
    save(name, 'label');
    
    mask = ones(size(label), 'uint8');

    name = strcat('./Edge/R-3/', Labels(index).name(1:end-4), '.mat');
    if ~exist(name, 'file')
        
        Contour3 = zeros(height, width,'double');
        if (nnz(contour_map)>20)    
            [Contour3, Thickness] = GenerateIDMask(contour_map, mask, 3);
        end
        save(name, 'Contour3');
        
    end
    
    name = strcat('./Weight/', Labels(index).name(1:end-4), '.mat');
    if exist(name, 'file')
        continue;
    end
    
    gt = imread(strcat('./Label/', Labels(index).name));
    contour = edge(gt, 'canny');

    SE = strel('disk', 5);
    range = imdilate(contour, SE);
    range(gt>0) = 0;

    [height, width] = size(gt);
    weights = zeros(height, width, 2, 'double');

    window_size = 10;
    for x = 1:height
        for y = 1:width
            if(range(x, y) > 0)
                top = max(x - window_size, 1);
                bottom = min(x + window_size, height);
                left = max(y - window_size, 1);
                right = min(y + window_size, width);
                Guidance = gt;
                for iter = 1:2
                    minRadius = 100;
                    minID = 0;
                    [r, ID] = FindNearest(Guidance(top:bottom, left:right), x - top + 1, y - left + 1 );
                    if (r < minRadius) && (ID > 0)
                        minRadius = r;
                        minID = ID;
                    end
                    if (minRadius < 100)
                        weights(x,y,iter) = minRadius;
                        Guidance(Guidance==minID) = 0;
                    else
                        break;
                    end
                end
            end
        end
    end

    Weights = weights(:,:,1)+weights(:,:,2);
    temp = weights(:,:,1);
    Weights(temp==0) = 0;
    temp = weights(:,:,2);
    Weights(temp==0) = 0;
    
    name = strcat('./Weight/', Labels(index).name(1:end-4), '.mat');
    save(name, 'Weights');
     
end
