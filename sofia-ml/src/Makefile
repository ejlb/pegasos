#================================================================================#
# Copyright 2009 Google Inc.                                                     #
#                                                                                # 
# Licensed under the Apache License, Version 2.0 (the "License");                #
# you may not use this file except in compliance with the License.               #
# You may obtain a copy of the License at                                        #
#                                                                                #
#      http://www.apache.org/licenses/LICENSE-2.0                                #
#                                                                                #
# Unless required by applicable law or agreed to in writing, software            #
# distributed under the License is distributed on an "AS IS" BASIS,              #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.       #
# See the License for the specific language governing permissions and            #
# limitations under the License.                                                 #
#================================================================================#

GCC= g++ -fpic -I/Library/Frameworks/Python.framework/Headers/ -I/usr/include/python2.7 -c 
LINK=g++ -shared -lm -lpython2.7 

#================================================================================#
#                           Main Make Commands                                   #
#================================================================================#

# Primary executable binary.
sofia-ml:
	python ../../generate_bindings.py > sofia_wrapper.c
	$(GCC) sofia-ml.cc 
	$(GCC) sofia-ml-methods.cc 
	$(GCC) sf-weight-vector.cc 
	$(GCC) sf-sparse-vector.cc 
	$(GCC) sf-data-set.cc 
	$(GCC) sf-hash-weight-vector.cc 
	$(GCC) sf-hash-inline.cc
	$(GCC) sofia_wrapper.c
	$(LINK) -o libsofia.so sofia-ml.o sofia-ml-methods.o sf-weight-vector.o sf-sparse-vector.o sf-data-set.o sf-hash-weight-vector.o sf-hash-inline.o
	$(GCC) sofia_wrapper.c
	g++ -shared -o sofia.so -L. -lsofia -lpython2.7 sofia_wrapper.o
	mv sofia.so libsofia.so ../../

# Remove all executable binaries (including tests).
clean:
	rm -f *.o
	rm -f sofia.so
	rm -f libsofia.so
