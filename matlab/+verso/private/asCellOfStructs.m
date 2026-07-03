function c = asCellOfStructs(x)
%ASCELLOFSTRUCTS Normalise a jsondecode'd JSON array to a cell array of scalar structs.
%   jsondecode returns a struct array when every element shares the same
%   fields, a cell array of structs otherwise, or a scalar struct for a
%   single-element array (or the array itself, decoded, for an empty JSON
%   array). This normalises all of those cases to a uniform cell array.
    if isempty(x)
        c = {};
    elseif iscell(x)
        c = x;
    elseif isstruct(x)
        c = num2cell(x);
    else
        error("verso:asCellOfStructs:unexpectedType", ...
            "Expected a JSON array of objects, got %s", class(x));
    end
end
