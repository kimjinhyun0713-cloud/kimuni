#!/bin/sh

sed "s/ratio: 0/ratio: 0.2/g" L8T00.yaml  > L8T02.yaml
sed "s/ratio: 0/ratio: 0.4/g" L8T00.yaml  > L8T04.yaml
sed "s/ratio: 0/ratio: 0.6/g" L8T00.yaml  > L8T06.yaml
sed "s/ratio: 0/ratio: 0.8/g" L8T00.yaml  > L8T08.yaml

sed "s/ratio: 0/ratio: 0.2/g" L5T00.yaml  > L5T02.yaml
sed "s/ratio: 0/ratio: 0.4/g" L5T00.yaml  > L5T04.yaml
sed "s/ratio: 0/ratio: 0.6/g" L5T00.yaml  > L5T06.yaml
sed "s/ratio: 0/ratio: 0.8/g" L5T00.yaml  > L5T08.yaml

sed "s/ratio: 0/ratio: 0.2/g" L2T00.yaml  > L2T02.yaml
sed "s/ratio: 0/ratio: 0.4/g" L2T00.yaml  > L2T04.yaml
sed "s/ratio: 0/ratio: 0.6/g" L2T00.yaml  > L2T06.yaml
sed "s/ratio: 0/ratio: 0.8/g" L2T00.yaml  > L2T08.yaml
