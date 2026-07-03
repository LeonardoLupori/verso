function vol = loadAtlasVolume(atlasDir)
%LOADATLASVOLUME Read a BrainGlobe atlas folder into an in-memory struct.
%   VOL = LOADATLASVOLUME(ATLASDIR) reads annotation.tiff, reference.tiff,
%   structures.json and metadata.json from ATLASDIR (the standard BrainGlobe
%   atlas cache layout -- see .claude/matlab-port.md) and returns:
%       Annotation   (AP, DV, LR) int32 -- region-ID volume
%       Reference    (AP, DV, LR) double -- grayscale template volume
%       RefScale     (1,1) double -- 255/max(Reference), for display scaling
%       ColorTable   containers.Map: region id (double) -> 1x3 uint8 RGB
%       ResolutionUm (1,1) double
%       Shape        (1,3) double, [AP, DV, LR]
%
%   Mirrors engine/atlas.py::AtlasVolume.__init__. Both volumes are loaded
%   eagerly (matching the Python constructor), so construct a
%   verso.VersoRegistration once and reuse it for repeated calls rather than
%   re-loading the atlas per call.
%
%   NOTE: tiffreadVolume (Image Processing Toolbox) returns a multi-page TIFF
%   as (rows, cols, pages); BrainGlobe/Python's tifffile.imread returns
%   (pages, rows, cols) = (AP, DV, LR). The permute below corrects for this --
%   see .claude/matlab-port.md before changing it.
    meta = jsondecode(fileread(fullfile(atlasDir, "metadata.json")));
    structuresRaw = asCellOfStructs(jsondecode(fileread(fullfile(atlasDir, "structures.json"))));

    annotation = permute(tiffreadVolume(fullfile(atlasDir, "annotation.tiff")), [3 1 2]);
    reference = permute(tiffreadVolume(fullfile(atlasDir, "reference.tiff")), [3 1 2]);

    annotation = int32(annotation);
    reference = double(reference);
    refMax = max(reference(:));
    if refMax > 0
        refScale = 255.0 / refMax;
    else
        refScale = 1.0;
    end

    colorTable = containers.Map("KeyType", "double", "ValueType", "any");
    colorTable(0) = uint8([0 0 0]);
    for k = 1:numel(structuresRaw)
        st = structuresRaw{k};
        rgb = double(st.rgb_triplet(:)');
        colorTable(double(st.id)) = uint8(rgb);
    end

    resolution = double(meta.resolution(:)');

    vol = struct( ...
        "Annotation", annotation, ...
        "Reference", reference, ...
        "RefScale", refScale, ...
        "ColorTable", colorTable, ...
        "ResolutionUm", resolution(1), ...
        "Shape", double(meta.shape(:)'));
end
