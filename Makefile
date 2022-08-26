yapsit: yapsit.c pokemon.h
	gcc -O2 -Wall -Wextra -Werror yapsit.c -o yapsit

pokemon.h: gen_header.py
	python3 gen_header.py > pokemon.h.bak
	mv pokemon.h.bak pokemon.h

.PHONY: clean
clean:
	rm -f pokemon.h yapsit

.PHONY: run
run: yapsit
	./yapsit
