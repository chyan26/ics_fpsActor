#!/bin/bash

steps=-25
count=40
exptime=0.8

oneCmd.py fps moveToHome all
oneCmd.py iic moveToPfsDesign designId=0x356557ee9790b9d

oneCmd.py mcs expose object exptime=$exptime doFibreID
for i in `seq $count`; do
    echo "starting loop $i of $count"

    oneCmd.py fps cobraMoveSteps theta stepsize=$steps
    oneCmd.py mcs expose object exptime=$exptime doFibreID
done