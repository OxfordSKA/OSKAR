classdef Type < uint32
    %TYPE Summary of this function goes here
    %   Detailed explanation goes here
    
    %oskar_type(bitor(oskar_type.Single, oskar_type.Complex))
    
    enumeration
        Single                  (1)        % 0x0001
        Double                  (2)        % 0x0002
        Int                     (4)        % 0x0004
        Complex                 (192)      % 0x00C0
        Matrix                  (1024)     % 0x0400
        Single_complex          (193)      % 0x00C1 == 0x0001 | 0x00C0
        Double_complex          (194)      % 0x00C2 == 0x0002 | 0x00C0
        Single_complex_matrix   (1217)     % 0x04C1 == 0x00C1 | 0x0400
        Double_complex_matrix   (1218)     % 0x04C2 == 0x00C2 | 0x0400
    end
    
end

