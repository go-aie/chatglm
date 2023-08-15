# ChatGLM

[![Go Reference](https://pkg.go.dev/badge/github.com/go-aie/chatglm/vulndb.svg)][1]

Go binding for [ChatGTLM.cpp][2].


## Usage

### Get the Code

```bash
git clone --recursive https://github.com/go-aie/chatglm.git && cd chatglm
```

Or:

```bash
git clone https://github.com/go-aie/chatglm.git && cd chatglm
git submodule update --init --recursive
```

### Quantize Model

Transform ChatGLM-6B into 4-bit quantized GGML format:

```bash
make convert
```

For other transformations, see [Quantize Model][3].

### Build & Test

Build the ChatGLM.cpp libraries:

```bash
make build
```

Run the tests:

```bash
go test -v -race ./...
```


## License

[MIT](LICENSE)


[1]: https://pkg.go.dev/github.com/go-aie/chatglm
[2]: https://github.com/li-plus/chatglm.cpp
[3]: https://github.com/li-plus/chatglm.cpp#getting-started