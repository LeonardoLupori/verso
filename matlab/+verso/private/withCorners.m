function ptsAll = withCorners(pts)
%WITHCORNERS Prepend the four invisible corner anchors to normalised points.
%   PTSALL = WITHCORNERS(PTS) prepends identity anchors at [-0.1,-0.1],
%   [1.1,-0.1], [-0.1,1.1], [1.1,1.1] -- placed 10% outside the frame so every
%   in-image pixel falls strictly inside the convex hull and border triangles
%   are interpolated rather than clamped (matching VisuAlign's triangulation).
%
%   Mirrors engine/warping.py::_with_corners exactly.
    corners = [-0.1 -0.1; 1.1 -0.1; -0.1 1.1; 1.1 1.1];
    ptsAll = [corners; pts];
end
