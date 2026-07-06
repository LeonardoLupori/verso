function out = warpPointsSectionToAtlas(pointsNorm, srcPx, dstPx, workW, workH)
%WARPPOINTSSECTIONTOATLAS Map section-space points into atlas-space via the Delaunay warp.
%   OUT = WARPPOINTSSECTIONTOATLAS(POINTSNORM, SRCPX, DSTPX, WORKW, WORKH)
%   triangulates on the *dst* (section) anchors and interpolates *src*
%   (atlas) coordinates for each input point. Points outside the convex
%   hull pass through unchanged (clipped to [0, 1]).
%
%   Mirrors engine/warping.py::warp_points_section_to_atlas exactly
%   (including the VisuAlign-parity aspect-ratio pre-scaling and the four
%   invisible corner anchors) -- see .claude/matlab-port.md.
%
%   POINTSNORM: (M, 2) normalised section-space points in [0, 1].
%   SRCPX, DSTPX: (N, 2) atlas-space / section-space control points, in
%       working-resolution pixels (may be empty, i.e. 0x2, for no warp).
%   WORKW, WORKH: working image dimensions in pixels (normalises control
%       points; does not affect POINTSNORM, already normalised).
%
%   OUT: (M, 2) normalised atlas-space points.
    pts = reshape(double(pointsNorm), [], 2);
    wh = [double(workW), double(workH)];

    srcPx = reshape(double(srcPx), [], 2);
    dstPx = reshape(double(dstPx), [], 2);
    srcNorm = srcPx ./ wh;
    dstNorm = dstPx ./ wh;
    aspect = workW / workH;

    out = min(max(pts, 0.0), 1.0);
    if isempty(dstNorm) || npAllClose(srcNorm, dstNorm)
        return
    end

    corners = [-0.1 -0.1; 1.1 -0.1; -0.1 1.1; 1.1 1.1];
    srcAll = [corners; srcNorm];
    dstAll = [corners; dstNorm];

    scale = [aspect, 1.0];
    DT = delaunayTriangulation(dstAll .* scale);
    [triId, bary] = pointLocation(DT, pts .* scale);

    valid = ~isnan(triId);
    if ~any(valid)
        return
    end

    vertIdx = DT.ConnectivityList(triId(valid), :);  % (Nvalid, 3)
    baryValid = bary(valid, :);                       % (Nvalid, 3)

    srcU = srcAll(:, 1);
    srcV = srcAll(:, 2);
    uVerts = srcU(vertIdx);  % linear indexing preserves shape: (Nvalid, 3)
    vVerts = srcV(vertIdx);  % (Nvalid, 3)
    outU = sum(baryValid .* uVerts, 2);
    outV = sum(baryValid .* vVerts, 2);

    out(valid, 1) = min(max(outU, 0.0), 1.0);
    out(valid, 2) = min(max(outV, 0.0), 1.0);
end
