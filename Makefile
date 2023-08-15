.PHONY: init
init:
	cd $$PWD/chatglm.cpp && git submodule update --init --recursive
	python3 -m venv venv
	$$PWD/venv/bin/pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

.PHONY: convert
convert: init
	$$PWD/venv/bin/python chatglm.cpp/chatglm_cpp/convert.py -i THUDM/chatglm-6b -t q4_0 -o chatglm-ggml.bin

.PHONY: build
build: init
	cd $$PWD/chatglm.cpp && cmake -B build && cmake --build build -j
