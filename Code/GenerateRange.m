function [SearchingMask] = GenerateRange( SearchingRadius, WightedRange )
% Function to generate the searching range of the skeleton map in the
% reference map
% Input:  SearchingRadius --> searching radius of each pixel in the reference skeleton
%                            map, the value of each pixel in the
%                            SearchingRadius is the radius of the pixel in
%                            the reference skeleton map
%         Mask -->  the FOV mask of the fundus image
% Output: SearchingMask -->  the mask denotes the searching range

[height, width] = size(SearchingRadius);
SearchingMask = zeros(height, width, 'uint8');
for x = 1:height
    for y = 1:width
        if( SearchingRadius(x, y) > 0 )
            
            top = max(x - 10, 1);
            bottom = min(x + 10, height);
            left = max(y - 10, 1);
            right = min(y + 10, width);
            
            for i = top:bottom
                for j = left:right
                    distance = sqrt(double((i - x)^2 + (j - y)^2));
                    if ((distance <= SearchingRadius(x,y))&&(distance<=WightedRange(i,j)))
                        SearchingMask(i,j) = 1;
                        WightedRange(i,j) = 0;
                    end
                end
            end
        end
    end
end
%SearchingMask = SearchingMask.*Mask;