function out = barycentricMap(triPts, valuePts, query, scale)
%BARYCENTRICMAP Piecewise-affine map of QUERY by barycentric interpolation.
%   OUT = BARYCENTRICMAP(TRIPTS, VALUEPTS, QUERY, SCALE) triangulates TRIPTS
%   (scaled to VisuAlign's aspect via SCALE), locates each QUERY point's
%   enclosing triangle, and interpolates the corresponding VALUEPTS with that
%   point's barycentric weights. The two warp directions differ only in which
%   set is triangulated and which is interpolated, so both call this with the
%   roles swapped.
%
%   Mirrors engine/warping.py::_barycentric_map exactly -- see
%   .claude/matlab-port.md. Uses MATLAB's pointLocation, which returns the
%   barycentric coordinates directly as its second output (NaN row outside the
%   hull), replacing scipy's Delaunay + tri.transform in one call.
%
%   TRIPTS, VALUEPTS: (N, 2) normalised points (corner anchors included),
%       aligned so VALUEPTS(k,:) is the value at TRIPTS(k,:).
%   QUERY: (M, 2) normalised points to map.
%   SCALE: (1, 2) anisotropy factor applied to the triangulation and queries.
%
%   OUT: (M, 2) interpolated values clipped to [0, 1]; query points outside the
%       convex hull pass through unchanged (clipped).
    out = min(max(query, 0.0), 1.0);

    DT = delaunayTriangulation(triPts .* scale);
    [triId, bary] = pointLocation(DT, query .* scale);

    valid = ~isnan(triId);
    if ~any(valid)
        return
    end

    vertIdx = DT.ConnectivityList(triId(valid), :);  % (Nvalid, 3)
    baryValid = bary(valid, :);                       % (Nvalid, 3)

    valX = valuePts(:, 1);
    valY = valuePts(:, 2);
    xVerts = valX(vertIdx);  % linear indexing preserves shape: (Nvalid, 3)
    yVerts = valY(vertIdx);  % (Nvalid, 3)
    outX = sum(baryValid .* xVerts, 2);
    outY = sum(baryValid .* yVerts, 2);

    out(valid, 1) = min(max(outX, 0.0), 1.0);
    out(valid, 2) = min(max(outY, 0.0), 1.0);
end
