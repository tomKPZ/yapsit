pokesprite: pokesprite.c pokemon.h
	gcc -O2 -Wall -Wextra -Werror pokesprite.c -o pokesprite
	strip pokesprite

pokemon.h: gen_header.py
	python3 gen_header.py > pokemon.h

.PHONY: clean
clean:
	rm -f pokemon.h pokesprite

.PHONY: run
run: pokesprite
	./pokesprite
