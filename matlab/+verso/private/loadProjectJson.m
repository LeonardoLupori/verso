function proj = loadProjectJson(jsonPath)
%LOADPROJECTJSON Parse project-verso.json into section snapshots for coordinate math.
%   PROJ = LOADPROJECTJSON(JSONPATH) reads a native VERSO project file and
%   returns a struct:
%       ResolutionUm  (1,1) double
%       AtlasShape    (1,3) double, [AP, DV, LR]
%       AtlasName     string
%       Ids           cellstr, section ids in project order
%       Snapshots     containers.Map: sectionId (char) -> snapshot struct
%
%   Snapshot struct fields (mirror engine/registration.py::_SectionSnapshot):
%       Id, OriginalPath, WorkW, WorkH, FullW, FullH,
%       O, U, V (1x3 each, atlas voxel space),
%       SrcPx, DstPx (Nx2 each, working-resolution pixel control points),
%       FlipH, FlipV (logical), Aligned (logical, non-degenerate plane)
%
%   Mirrors engine/registration.py::VersoRegistration._init_from_project and
%   ::_build_snapshot. Note: control points (ControlPoint.src_x/y, dst_x/y)
%   are stored in the JSON as **working-resolution pixel coordinates**, not
%   normalised [0,1] -- see .claude/data-model.md ("Control point format").
    raw = jsondecode(fileread(jsonPath));

    atlas = raw.atlas;
    resolutionUm = double(atlas.resolution_um);
    atlasShape = double(atlas.shape(:)');
    if resolutionUm <= 0 || any(atlasShape <= 0)
        error("verso:loadProjectJson:incompleteAtlasMetadata", ...
            "Project atlas metadata is incomplete (resolution_um / shape); " + ...
            "the project file is not self-contained for coordinate math.");
    end

    sectionsRaw = asCellOfStructs(raw.sections);

    ids = cell(1, numel(sectionsRaw));
    snapshots = containers.Map("KeyType", "char", "ValueType", "any");
    for k = 1:numel(sectionsRaw)
        snap = iBuildSnapshot(sectionsRaw{k});
        ids{k} = snap.Id;
        snapshots(snap.Id) = snap;
    end

    proj = struct( ...
        "ResolutionUm", resolutionUm, ...
        "AtlasShape", atlasShape, ...
        "AtlasName", string(atlas.name), ...
        "Ids", {ids}, ...
        "Snapshots", snapshots);
end

function snap = iBuildSnapshot(s)
    workWH = double(s.resolution_thumbnail_wh(:)');
    fullWH = double(s.resolution_original_wh(:)');
    workW = workWH(1); workH = workWH(2);
    fullW = fullWH(1); fullH = fullWH(2);
    if min([workW, workH, fullW, fullH]) <= 0
        error("verso:loadProjectJson:unpopulatedDimensions", ...
            "Section %s has unpopulated pixel dimensions; the project " + ...
            "file is not self-contained for coordinate math.", string(s.id));
    end

    [o, u, v] = anchoringToVectors(s.alignment.anchoring);

    cps = asCellOfStructs(s.warp.control_points);
    n = numel(cps);
    srcPx = zeros(n, 2);
    dstPx = zeros(n, 2);
    for k = 1:n
        srcPx(k, :) = [double(cps{k}.src_x), double(cps{k}.src_y)];
        dstPx(k, :) = [double(cps{k}.dst_x), double(cps{k}.dst_y)];
    end

    aligned = norm(cross(u, v)) > 0.0;

    snap = struct( ...
        "Id", char(string(s.id)), ...
        "OriginalPath", char(string(s.original_path)), ...
        "WorkW", workW, "WorkH", workH, ...
        "FullW", fullW, "FullH", fullH, ...
        "O", o, "U", u, "V", v, ...
        "SrcPx", srcPx, "DstPx", dstPx, ...
        "FlipH", logical(s.preprocessing.flip_horizontal), ...
        "FlipV", logical(s.preprocessing.flip_vertical), ...
        "Aligned", aligned);
end
