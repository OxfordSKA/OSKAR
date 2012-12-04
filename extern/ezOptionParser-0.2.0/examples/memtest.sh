#!/bin/bash

test=parseindex
valgrind --tool=memcheck --leak-check=full --show-reachable=yes --log-file=tmp.valgrind.log ./$test a b c d e f g -f ooz -c bar,baz -l 1:-9:0 h i j k -c opto,mum -f foo bub > /dev/null
grep "leak" tmp.valgrind.log || echo "ERROR: Leaks are possible. See tmp.$test.log" && mv tmp.valgrind.log tmp.$test.log

test=pretty
valgrind --tool=memcheck --leak-check=full --show-reachable=yes --log-file=tmp.valgrind.log ./$test a b c d e f g -p -foo -bar -l 1:-9:0 h i j k  > /dev/null
grep "leak" tmp.valgrind.log || echo "ERROR: Leaks are possible. See tmp.$test.log" && mv tmp.valgrind.log tmp.$test.log

test=full
valgrind --tool=memcheck --leak-check=full --show-reachable=yes  --log-file=tmp.valgrind.log ./$test -i in2.txt > /dev/null
grep "leak" tmp.valgrind.log || echo "ERROR: Leaks are possible. See tmp.$test.log" && mv tmp.valgrind.log tmp.$test.log

test=complete
valgrind --tool=memcheck --leak-check=full --show-reachable=yes --log-file=tmp.valgrind.log ./$test a b -f --list 1,2,3 --list 4,5,6,7,8 -s string -int -2147483648,2147483647 -ulong 9223372036854775807 -float 3.40282e+038 -double 1.79769e+308 f1 f2 f3 f4 f5 f6 fout > /dev/null
grep "leak" tmp.valgrind.log || echo "ERROR: Leaks are possible. See tmp.$test.log" && mv tmp.valgrind.log tmp.$test.log

test=valid
valgrind --tool=memcheck --leak-check=full --show-reachable=yes --log-file=tmp.valgrind.log ./$test -i validin1.txt > /dev/null
grep "leak" tmp.valgrind.log || echo "ERROR: Leaks are possible. See tmp.$test.log" && mv tmp.valgrind.log tmp.$test.log

test=validfast
valgrind --tool=memcheck --leak-check=full --show-reachable=yes --log-file=tmp.valgrind.log ./$test -i validin2.txt > /dev/null
grep "leak" tmp.valgrind.log || echo "ERROR: Leaks are possible. See tmp.$test.log" && mv tmp.valgrind.log tmp.$test.log

test=validrange
valgrind --tool=memcheck --leak-check=full --show-reachable=yes --log-file=tmp.valgrind.log ./$test -i validin2.txt > /dev/null
grep "leak" tmp.valgrind.log || echo "ERROR: Leaks are possible. See tmp.$test.log" && mv tmp.valgrind.log tmp.$test.log

test=validrangefast
valgrind --tool=memcheck --leak-check=full --show-reachable=yes --log-file=tmp.valgrind.log ./$test -i validin2.txt > /dev/null
grep "leak" tmp.valgrind.log || echo "ERROR: Leaks are possible. See tmp.$test.log" && mv tmp.valgrind.log tmp.$test.log
