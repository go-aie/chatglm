package chatglm_test

import (
	"fmt"

	"github.com/go-aie/chatglm"
)

func Example_generate() {
	// Run `make convert` to get the quantized GGML model.
	p := chatglm.New("./chatglm-ggml.bin")
	defer p.Delete()

	output := p.Generate("你好")
	fmt.Println(output)

	// Output:
	// 你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。
}

func Example_streamGenerate() {
	// Run `make convert` to get the quantized GGML model.
	p := chatglm.New("./chatglm-ggml.bin")
	defer p.Delete()

	history := []*chatglm.Turn{{Question: "你好", Answer: "你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。"}}
	output := p.StreamGenerate(chatglm.BuildPrompt("晚上睡不着应该怎么办", history), chatglm.WithDoSample(false))
	for text := range output {
		fmt.Print(text)
		//fmt.Printf("%s|", text)
	}

	// Output:
	// 晚上睡不着可能会让人感到焦虑和不安，但以下是一些可能有用的建议：
	//
	// 1. 保持放松：尝试进行深呼吸、渐进性肌肉松弛或冥想等放松技巧，帮助放松身体和头脑。
	//
	// 2. 避免使用电子设备：电子设备的蓝光可能会影响睡眠，因此建议在睡前几个小时停止使用电子设备。
	//
	// 3. 创造一个舒适的睡眠环境：保持房间安静、凉爽、黑暗和舒适，有助于更容易入睡。
	//
	// 4. 避免饮用含咖啡因的饮料：咖啡因可能会影响睡眠，因此建议在睡前几个小时避免饮用含咖啡因的饮料。
	//
	// 5. 规律作息：保持规律的作息习惯，例如每天在相同的时间上床睡觉和起床，有助于身体更容易地适应规律的睡眠模式。
	//
	// 如果这些方法不能帮助入睡，建议咨询医生或睡眠专家，获取更专业的建议和帮助。
}
