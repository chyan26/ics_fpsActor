#!/bin/bash

steps=-40
count=40
exptime=0.8

oneCmd.py fps moveToHome all
oneCmd.py iic moveToPfsDesign designId=0x4993e56d9549f2f9

oneCmd.py mcs expose object exptime=$exptime doFibreID
for i in `seq $count`; do
    echo "starting loop $i of $count"
    
    
    oneCmd.py fps makeMotorMap phi stepsize=$steps totalsteps=1600 repeat=1 slowOnly
    #oneCmd.py fps cobraMoveSteps phi stepsize=$steps
    #oneCmd.py mcs expose object exptime=$exptime doFibreID
done