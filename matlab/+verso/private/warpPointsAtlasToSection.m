function out = warpPointsAtlasToSection(pointsNorm, srcPx, dstPx, workW, workH)
%WARPPOINTSATLASTOSECTION Map atlas-space points into section-space via the Delaunay warp.
%   OUT = WARPPOINTSATLASTOSECTION(POINTSNORM, SRCPX, DSTPX, WORKW, WORKH)
%   is the mirror of WARPPOINTSSECTIONTOATLAS: triangulates on the *src*
%   (atlas) anchors and interpolates *dst* (section) coordinates. Points
%   outside the convex hull pass through unchanged (clipped to [0, 1]).
%
%   Mirrors engine/warping.py::warp_points_atlas_to_section exactly -- see
%   .claude/matlab-port.md.
%
%   POINTSNORM: (M, 2) normalised atlas-space points in [0, 1].
%   SRCPX, DSTPX: (N, 2) atlas-space / section-space control points, in
%       working-resolution pixels (may be empty, i.e. 0x2, for no warp).
%   WORKW, WORKH: working image dimensions in pixels.
%
%   OUT: (M, 2) normalised section-space points.
    pts = reshape(double(pointsNorm), [], 2);
    wh = [double(workW), double(workH)];

    srcPx = reshape(double(srcPx), [], 2);
    dstPx = reshape(double(dstPx), [], 2);
    srcNorm = srcPx ./ wh;
    dstNorm = dstPx ./ wh;
    aspect = workW / workH;

    out = min(max(pts, 0.0), 1.0);
    if isempty(srcNorm) || npAllClose(srcNorm, dstNorm)
        return
    end

    corners = [-0.1 -0.1; 1.1 -0.1; -0.1 1.1; 1.1 1.1];
    srcAll = [corners; srcNorm];
    dstAll = [corners; dstNorm];

    scale = [aspect, 1.0];
    DT = delaunayTriangulation(srcAll .* scale);
    [triId, bary] = pointLocation(DT, pts .* scale);

    valid = ~isnan(triId);
    if ~any(valid)
        return
    end

    vertIdx = DT.ConnectivityList(triId(valid), :);  % (Nvalid, 3)
    baryValid = bary(valid, :);                       % (Nvalid, 3)

    dstX = dstAll(:, 1);
    dstY = dstAll(:, 2);
    xVerts = dstX(vertIdx);  % linear indexing preserves shape: (Nvalid, 3)
    yVerts = dstY(vertIdx);  % (Nvalid, 3)
    outX = sum(baryValid .* xVerts, 2);
    outY = sum(baryValid .* yVerts, 2);

    out(valid, 1) = min(max(outX, 0.0), 1.0);
    out(valid, 2) = min(max(outY, 0.0), 1.0);
end
