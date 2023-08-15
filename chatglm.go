package chatglm

/*
#cgo CXXFLAGS: -I${SRCDIR}/chatglm.cpp -I${SRCDIR}/chatglm.cpp/third_party/ggml/include/ggml -I${SRCDIR}/chatglm.cpp/third_party/sentencepiece/src
#cgo LDFLAGS: -L${SRCDIR}/chatglm.cpp/build/lib -lchatglm -lggml -lsentencepiece
#cgo CXXFLAGS: -std=c++17
#cgo darwin LDFLAGS: -framework Accelerate

#include "binding.h"
#include <stdlib.h>
*/
import "C"

import (
	"bytes"
	"sync"
	"text/template"
	"unsafe"
)

type GenerateOptions struct {
	MaxLength         int
	MaxContextLength  int
	DoSample          bool
	TopK              int
	TopP              float32
	Temperature       float32
	RepetitionPenalty float32
	NumThreads        int
}

func newGenerateOptions(options ...GenerateOption) *GenerateOptions {
	opts := &GenerateOptions{
		MaxLength:         2048,
		MaxContextLength:  512,
		DoSample:          true,
		TopK:              0,
		TopP:              0.7,
		Temperature:       0.95,
		RepetitionPenalty: 1.0,
		NumThreads:        0,
	}
	for _, opt := range options {
		opt(opts)
	}
	return opts
}

// newGenerationConfig creates a new instance of the C struct GenerationConfig.
func (opts *GenerateOptions) newGenerationConfig() (*C.GenerationConfig, func()) {
	conf := C.NewGenerationConfig(
		C.int(opts.MaxLength),
		C.int(opts.MaxContextLength),
		C.bool(opts.DoSample),
		C.int(opts.TopK),
		C.float(opts.TopP),
		C.float(opts.Temperature),
		C.float(opts.RepetitionPenalty),
		C.int(opts.NumThreads),
	)
	return conf, func() { C.DeleteGenerationConfig(conf) }
}

type GenerateOption func(opts *GenerateOptions)

func WithMaxLength(v int) GenerateOption {
	return func(opts *GenerateOptions) { opts.MaxLength = v }
}

func WithMaxContextLength(v int) GenerateOption {
	return func(opts *GenerateOptions) { opts.MaxContextLength = v }
}

func WithDoSample(v bool) GenerateOption {
	return func(opts *GenerateOptions) { opts.DoSample = v }
}

func WithTopK(v int) GenerateOption {
	return func(opts *GenerateOptions) { opts.TopK = v }
}

func WithTopP(v float32) GenerateOption {
	return func(opts *GenerateOptions) { opts.TopP = v }
}

func WithTemperature(v float32) GenerateOption {
	return func(opts *GenerateOptions) { opts.Temperature = v }
}

func WithRepetitionPenalty(v float32) GenerateOption {
	return func(opts *GenerateOptions) { opts.RepetitionPenalty = v }
}

func WithNumThreads(v int) GenerateOption {
	return func(opts *GenerateOptions) { opts.NumThreads = v }
}

type Turn struct {
	Question string
	Answer   string
}

func BuildPrompt(query string, history []*Turn) string {
	text := `
{{- range $i, $turn := $.History -}}
[Round {{$i}}]

问：{{$turn.Question}}

答：{{$turn.Answer}}

{{end -}}
[Round {{len $.History}}]

问：{{$.Query}}

答：`
	tmpl, err := template.New("").Parse(text)
	if err != nil {
		panic(err)
	}

	data := struct {
		Query   string
		History []*Turn
	}{
		Query:   query,
		History: history,
	}
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		panic(err)
	}

	return buf.String()
}

type ChatGLM struct {
	p *C.Pipeline
}

func New(path string) *ChatGLM {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	return &ChatGLM{
		p: C.NewPipeline(cpath),
	}
}

func (c *ChatGLM) Delete() {
	C.DeletePipeline(c.p)
}

// Generate generates a response according to the given prompt.
func (c *ChatGLM) Generate(prompt string, options ...GenerateOption) string {
	cprompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cprompt))

	opts := newGenerateOptions(options...)
	conf, cleanup := opts.newGenerationConfig()
	defer cleanup()

	length := opts.MaxLength
	if length == 0 {
		length = 10
	}
	output := make([]byte, length)
	pout := (*C.char)(unsafe.Pointer(&output[0]))

	C.Pipeline_Generate(c.p, cprompt, conf, pout)
	return C.GoString(pout)
}

// StreamGenerate is like Generate but operates in stream mode.
func (c *ChatGLM) StreamGenerate(prompt string, options ...GenerateOption) <-chan string {
	ch := make(chan string)
	setCallback(unsafe.Pointer(c.p), channel{C: ch}.Send)

	go func() {
		cprompt := C.CString(prompt)
		defer C.free(unsafe.Pointer(cprompt))

		opts := newGenerateOptions(options...)
		conf, cleanup := opts.newGenerationConfig()
		defer cleanup()

		C.Pipeline_Generate(c.p, cprompt, conf, nil)
		setCallback(unsafe.Pointer(c.p), nil)
	}()

	return ch
}

// CGo only allows us to use static calls from C to Go, so we register the
// callbacks in this map and call streamCallback from the C code.
var (
	callbacks sync.Map // uintptr => func(string, bool)
)

//export streamCallback
func streamCallback(pipelinePtr unsafe.Pointer, text *C.char, end C.int) {
	if callback, ok := callbacks.Load(uintptr(pipelinePtr)); ok {
		fn := callback.(func(string, bool))
		fn(C.GoString(text), int(end) == 1)
	}
}

// setCallback registers a stream callback for ChatGLM. Pass in a nil callback to
// remove the callback.
func setCallback(pipelinePtr unsafe.Pointer, callback func(string, bool)) {
	if callback == nil {
		callbacks.Delete(uintptr(pipelinePtr))
	} else {
		callbacks.Store(uintptr(pipelinePtr), callback)
	}
}

// channel is a Channel-based implementation of the callback function.
type channel struct {
	C chan<- string
}

func (c channel) Send(text string, stop bool) {
	if text != "" {
		c.C <- text
	}
	if stop {
		close(c.C)
	}
}
