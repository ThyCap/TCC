// Gmsh project created on Wed Jun 28 08:48:24 2023
//+
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 1, 1, 1};
//+
Physical Surface("XY-", 13) = {5};
//+
Physical Surface("XY+", 14) = {6};
//+
Physical Surface("XZ-", 15) = {3};
//+
Physical Surface("XZ+", 16) = {4};
//+
Physical Surface("YZ-", 17) = {1};
//+
Physical Surface("YZ+", 18) = {2};
//+
Physical Volume("cube", 19) = {1};
