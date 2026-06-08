SHELL := /bin/bash

.PHONY: all lib tests test clean

all: lib tests

lib:
	$(MAKE) -C lib

tests:
	$(MAKE) -C tests

test:
	$(MAKE) -C tests test

clean:
	$(MAKE) -C tests clean
	$(MAKE) -C lib clean
