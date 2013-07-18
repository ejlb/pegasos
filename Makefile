all:
	cd sofia-ml/src; make
	ipython ./tests.py

clean:
	cd sofia-ml/src; make clean
	rm -f sofia.so
	rm -f libsofia.so
