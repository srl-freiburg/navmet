NOSETESTS ?= nosetests

all:
	python setup.py build_ext -i

coverage:
	nosetests --with-coverage --cover-html --cover-package=navmet navmet

test-code: all
	$(NOSETESTS) -s -v navmet


clean:
	find | grep .pyc | xargs rm

test-doc:
	$(NOSETESTS) -s -v doc/*.rst doc/modules/

test: test-code test-doc
