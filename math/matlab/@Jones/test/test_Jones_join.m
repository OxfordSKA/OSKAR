import oskar.Jones

a1 = [[1 3]; [2 4]];
a2 = [[5 7]; [6 8]];

% Test scalar jones join with memory on the CPU.
J1 = Jones(a1, 'scalar');
J2 = Jones(a2, 'scalar');
% J3 = J1 * J2
J3 = Jones.join(J1, J2);
expected = a1 .* a2;
values   = J3.values;
for i=1:length(expected(:))
    assert(expected(i) == values(i), 'Scalar jones join failed!');
end
clear J1 J2 J3 expected values i;

% Test Matrix jones join with memory on the CPU.
J1 = Jones(a1, 'matrix');
J2 = Jones(a2, 'matrix');
% J3 = J1 * J2
J3 = Jones.join(J1, J2);
expected = a1 * a2;
values   = J3.values;
for i=1:length(expected(:))
    assert(expected(i) == values(i), 'Matrix jones join failed (cpu memory)!');
end
clear J1 J2 J3 expected values i;

% Test Matrix jones join with memory on the GPU.
J1 = Jones(a1, 'matrix', 'gpu');
J2 = Jones(a2, 'matrix', 'gpu');
J3 = Jones.join(J1, J2);
expected = a1 * a2;
values   = J3.values;
for i=1:length(expected(:))
    assert(expected(i) == values(i), 'Matrix jones join failed (gpu memory)!');
end
clear J1 J2 J3 expected values i;

% Test Matrix jones join from right
J1 = Jones(a1, 'matrix');
J2 = Jones(a2, 'matrix');
% this = this * other ===> J1 = J1 * J2
J1.join_from_right(J2);
expected = a1 * a2;
values   = J1.values;
for i=1:length(expected(:))
    assert(expected(i) == values(i), 'Matrix jones join from right');
end
clear J1 J2 expected values i;

% Test Matrix jones join to left
this  = Jones(a1, 'matrix', 'cpu');
other = Jones(a2, 'matrix', 'gpu');
% other = other * this
oskar.Jones_join_to_left(this.pointer, other.pointer); % Note not a class function yet...
expected = a2 * a1;
values = other.values;
for i=1:length(expected(:))
    assert(expected(i) == values(i), 'Matrix jones join to left');
end
clear this other expected values i;

clear a1 a2

disp('SUCCESS!! All oskar.Jones tests passed! ');

disp('"what oskar" returns:');
what oskar
