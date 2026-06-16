#!/bin/sh

sed "s/MCL: 2/MCL: 4/g" L02.yaml > L04.yaml
sed "s/MCL: 4/MCL: 8/g" L04.yaml > L08.yaml
sed "s/MCL: 4/MCL: 10/g" L04.yaml > L10.yaml
sed "s/MCL: 4/MCL: 12/g" L04.yaml > L12.yaml
