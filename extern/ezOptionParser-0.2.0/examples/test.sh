#!/bin/bash 

# Search array if contains element.
# elementExists 1 "1 2 3"
# echo $?
function elementExists {
	local arr=$2;
	if [ -z "$1" ]; then return; fi;
	for i in ${arr[@]}; do
		if [ $i == $1 ]; then return 1; fi;
	done;
	return 0;
}

# Allow user to run specific tests only, such as: test.sh "1 10 11"
if [ -z "$1" ]; then
	NTESTS=58
	TESTS=`seq 1 1 $NTESTS`
else
	TESTS="$1"
	NTESTS=`echo "$1" | wc -w`
fi

if [ -z $TMPDIR ]; then
	TMPDIR="."
fi

testctr=0

####################################################################
elementExists 1 "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test 1..."
	./short 2>&1 | grep "Expected 2 arguments" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi
####################################################################
elementExists 2 "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test 2..."
	./short a 2>&1 | grep "but got 1" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi
####################################################################
elementExists 3 "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test 3..."
	./short a b 2>&1 | grep "Second file: b" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi
####################################################################
elementExists 4 "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test 4..."
	./short a -d b 2>&1 | grep "but got 1" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi
####################################################################
elementExists 5 "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test 5..."
	./short -d a b 2>&1 | grep "\-d was set" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi
####################################################################
elementExists 6 "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test 6..."
	./long -debug a b 2>&1 | grep "\-d was set" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi
####################################################################
elementExists 7 "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test 7..."
	./long --dbg a b 2>&1 | grep "\-d was set" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi
####################################################################
elementExists 8 "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test 8..."
	./long --debug a b 2>&1 | grep "\-d was set" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi
####################################################################
elementExists 9 "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test 9..."
	./multi 2>&1 | grep "Missing required" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi
####################################################################
CUR=10
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./multi -p 2>&1 | grep "unexpected number of arguments" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi
####################################################################
CUR=11
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./multi -p 1,2 2>&1 | grep "unexpected number of arguments" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi
####################################################################
CUR=12
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./multi -p 1,2,3 --point -1,1e6,-.314 2>&1 | grep "Point 1: \-1" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi
####################################################################
CUR=13
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./pretty -p 2>&1 | grep "Not set" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi
####################################################################
CUR=14
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./pretty -p -l 1:-9:0 2>&1 | grep "List: 1 -9 0" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi
####################################################################
CUR=15
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./pretty a b -p -foo -l 1:-9:0 c d 2>&1 | grep "1: \-foo" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi
####################################################################
CUR=16
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./pretty a b -p -foo -l 1:-9:0 c d 2>&1 | grep "3: b" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=17
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./full -i in2.txt 2>&1 > tmp.$CUR
	cmp test$CUR.truth tmp.$CUR 2>&1 > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=18
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./full -i in2.txt -d 987654321 -dbl 98.123e-120 +d +e tmp.$CUR 2>&1 > /dev/null
	cmp test$CUR.truth tmp.$CUR 2>&1 > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=19
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./full -d 987654321 -dbl 98.123e-120 +d +e tmp.$CUR 2>&1 > /dev/null
	cmp test$CUR.truth tmp.$CUR 2>&1 > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=20
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./full +i in3.txt -d 987654321 -dbl 98.123e-120 -n 2020202 +d +e tmp.willnotbecreated 2>&1 > /dev/null
	cmp test$CUR.truth tmp.$CUR 2>&1 > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=21
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -i validin1.txt -o tmp.$CUR 2>&1 > /dev/null
	cmp test$CUR.truth tmp.$CUR 2>&1 > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=22
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -b 2 2>&1 | grep "Got invalid argument \"2\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=23
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -b fals 2>&1 | grep "Got invalid argument \"fals\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=24
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -s1 -200 2>&1 | grep "Got invalid argument \"\-200\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=25
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -s1 200 2>&1 | grep "Got invalid argument \"200\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=26
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -s1list 0,127,200 2>&1 | grep "Got invalid argument \"200\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=27
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -u1 -300 2>&1 | grep "Got invalid argument \"\-300\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=28
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -u1 300 2>&1 | grep "Got invalid argument \"300\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=29
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -u1list 0,255,300 2>&1 | grep "Got invalid argument \"300\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=30
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -s2 -44444 2>&1 | grep "Got invalid argument \"\-44444\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=31
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -s2 44444 2>&1 | grep "Got invalid argument \"44444\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=32
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -s2list -32768,0,32767,44444 2>&1 | grep "Got invalid argument \"44444\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=33
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -u2 -1 2>&1 | grep "Got invalid argument \"\-1\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=34
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -u2 75535 2>&1 | grep "Got invalid argument \"75535\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=35
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -u2list 0,34567,65535,98765 2>&1 | grep "Got invalid argument \"98765\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=36
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -s4 -3147483648 2>&1 | grep "Got invalid argument \"\-3147483648\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=37
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -s4 3147483647 2>&1 | grep "Got invalid argument \"3147483647\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=38
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -s4list -2147483648,0,9147483647 2>&1 | grep "Got invalid argument \"9147483647\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=39
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -u4 -1 2>&1 | grep "Got invalid argument \"\-1\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=40
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -u4 4294967299 2>&1 | grep "Got invalid argument \"4294967299\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=41
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -u4list 0,4294967299 2>&1 | grep "Got invalid argument \"4294967299\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=42
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -u8 -9 2>&1 | grep "Got invalid argument \"\-9\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=43
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -u8 18446744073709551619 2>&1 | grep "Got invalid argument \"18446744073709551619\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=44
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -u8list 0,10,18446744073709551619 2>&1 | grep "Got invalid argument \"18446744073709551619\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=45
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -s8 -9223372036854775809 2>&1 | grep "Got invalid argument \"\-9223372036854775809\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=46
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -s8 -9223372036854775809 2>&1 | grep "Got invalid argument \"\-9223372036854775809\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=47
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -s8list 9223372036854775807,-9223372036854775809 2>&1 | grep "Got invalid argument \"\-9223372036854775809\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=48
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -f -3.40283e+038 2>&1 | grep "Got invalid argument \"\-3.40283e+038\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=49
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -f 3.40283e+038 2>&1 | grep "Got invalid argument \"3.40283e+038\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=50
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -flist 1e10,2.34,3.40283e+038 2>&1 | grep "Got invalid argument \"3.40283e+038\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=51
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -d -1.7977e+308 2>&1 | grep "Got invalid argument \"\-1.7977e+308\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=52
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -d 1.7977e+308 2>&1 | grep "Got invalid argument \"1.7977e+308\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=53
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./valid -dlist 1e10,-.3,1.7977e+308 2>&1 | grep "Got invalid argument \"1.7977e+308\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=54
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./validfast -i validin1.txt -o tmp.$CUR 2>&1 > /dev/null
	cmp test$CUR.truth tmp.$CUR 2>&1 > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=55
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./validrange -i validin2.txt -o tmp.$CUR 2>&1 > /dev/null
	cmp test$CUR.truth tmp.$CUR 2>&1 > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=56
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./validrangefast -i validin2.txt -o tmp.$CUR 2>&1 > /dev/null
	cmp test$CUR.truth tmp.$CUR 2>&1 > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=57
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./validrangefast -u8list 0,10,18446744073709551619 2>&1 | grep "Got invalid argument \"18446744073709551619\"" > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
CUR=58
elementExists $CUR "$TESTS"
if [ $? -eq 1 ]; then
	echo -n "Test $CUR..."
	./parseindex skip -f foo -c bar,baz -c bam -f ooz -c yum ignore 2>&1 | tr -d "\n" | grep '\-c indices\: 4 6 10 \-f indices: 2 8' > /dev/null
	if [ "$?" -ne "0" ]; then
		echo FAILED
	else
		echo PASSED
		testctr=$((testctr+1))
	fi
fi

####################################################################
echo $testctr of $NTESTS tests passed.