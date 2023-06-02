#!/bin/bash
rm -rf /data/dataset

mkdir /data/dataset -p
mkdir /data/dataset/images -p
mkdir /data/dataset/images/train -p
mkdir /data/dataset/images/val -p
mkdir /data/dataset/labels -p
mkdir /data/dataset/labels/train -p
mkdir /data/dataset/labels/val -p

tree /data/dataset