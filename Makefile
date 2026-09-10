IMAGE_NAME := model-frame-vector

DUMMY := $(shell git submodule update --init 1>&2)
include buildscripts/Makefile.tagger-model
