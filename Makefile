all:
	cd sofia-ml/src; make

clean:
	cd sofia-ml/src; make clean
	rm -f sofia.so
	rm -f libsofia.so
