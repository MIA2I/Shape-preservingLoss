function [ IDMask ] = GenerateIDMask( RefVessels, Mask, Levels )

%Hyperparameters
minLength = 16; % the predefined minimum length of the skeleton segment
avgLength = 16; % the predefined average length of the skeleton segment

%Initialization
Mask(Mask>0) = 1;
[height, width] = size(RefVessels);
RefVessels = uint8(RefVessels);
RefVessels(RefVessels>0) = 1;

[ RefThickness, RefminRadius, RefmaxRadius ] = CalcThickness( RefVessels, RefVessels);
RefSkeleton = RefThickness;
RefSkeleton(RefSkeleton~=1) = 0;
SearchingRadius = RefSkeleton * Levels;

% Segment the target skeleton map
[ SegmentID ] = SegmentSkeleton( RefSkeleton, minLength, avgLength );
SegmentID(Mask==0) = 0;

SearchingFOV = 255*ones(height, width, 'uint8');
SearchingMask = GenerateRange(SearchingRadius, SearchingFOV);
SearchingMask = SearchingMask.*(1-RefVessels);

[ WightedRange, RefminRadius, RefmaxRadius ] = CalcThickness( SearchingMask, 1-RefVessels);
WightedRange(WightedRange==0) = 255;

% Calculate the skeletal similarity for each segment
IDMask = zeros(height, width,'double');
for Index = 1:max(max(SegmentID))
    
    SegmentRadius = SearchingRadius;
    SegmentRadius(SegmentID~=Index) = 0;
    SegmentMask = GenerateRange(SegmentRadius, WightedRange);
    
    IDMask(SegmentMask>0) = Index;
    IDMask(SegmentID==Index) = -Index;
    
end
