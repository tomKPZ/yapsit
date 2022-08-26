pokesprite: pokesprite.c pokemon.h
	gcc -O2 -Wall -Wextra -Werror pokesprite.c -o pokesprite

pokemon.h: gen_header.py
	python3 gen_header.py > pokemon.h.bak
	mv pokemon.h.bak pokemon.h

.PHONY: clean
clean:
	rm -f pokemon.h pokesprite

.PHONY: run
run: pokesprite
	./pokesprite
