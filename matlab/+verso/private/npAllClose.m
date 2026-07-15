function tf = npAllClose(a, b)
%NPALLCLOSE Elementwise closeness check matching numpy.allclose's default tolerances.
%   TF = NPALLCLOSE(A, B) returns true iff ALL(ABS(A - B) <= ATOL + RTOL *
%   ABS(B)) with RTOL = 1e-5, ATOL = 1e-8 (numpy.allclose defaults), and A, B
%   are the same size.
    if ~isequal(size(a), size(b))
        tf = false;
        return
    end
    rtol = 1e-5;
    atol = 1e-8;
    tf = all(abs(a - b) <= (atol + rtol * abs(b)), "all");
end
