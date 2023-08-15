#include "chatglm.h"

#include "binding.h"
#include <vector>
#include <algorithm>

/* ===== structure/class definitions. ===== */

struct GenerationConfig : public chatglm::GenerationConfig {
    GenerationConfig(
        int max_length,
        int max_context_length,
        bool do_sample,
        int top_k,
        float top_p,
        float temperature,
        float repetition_penalty,
        int num_threads
    ): chatglm::GenerationConfig(max_length, max_context_length, do_sample, top_k, top_p,
                                 temperature, repetition_penalty, num_threads) {}
    ~GenerationConfig() {}
};

struct Pipeline : public chatglm::Pipeline {
    Pipeline(char* path): chatglm::Pipeline(path) {}
    ~Pipeline() {}
};

// CallbackStreamer is much like chatglm::TextStreamer except that it sends the
// generated text to a callback function, which is implemented in Go.
class CallbackStreamer : public chatglm::BaseStreamer {
  public:
    CallbackStreamer(Pipeline* pipeline, chatglm::BaseTokenizer *tokenizer)
        : pipeline_(pipeline), tokenizer_(tokenizer), is_prompt_(true), print_len_(0) {}
    void put(const std::vector<int> &output_ids) override;
    void end() override;

  private:
    Pipeline* pipeline_;
    chatglm::BaseTokenizer *tokenizer_;
    bool is_prompt_;
    std::vector<int> token_cache_;
    int print_len_;
};

/* ===== function implementations. ===== */

GenerationConfig* NewGenerationConfig(
    int max_length,
    int max_context_length,
    bool do_sample,
    int top_k,
    float top_p,
    float temperature,
    float repetition_penalty,
    int num_threads
) {
    return new GenerationConfig(max_length,
                                  max_context_length,
                                  do_sample,
                                  top_k,
                                  top_p,
                                  temperature,
                                  repetition_penalty,
                                  num_threads);
}

void DeleteGenerationConfig(GenerationConfig* p) {
    delete p;
}

Pipeline* NewPipeline(char* path) {
    return new Pipeline(path);
}

void DeletePipeline(Pipeline* p) {
    delete p;
}

void Pipeline_Generate(Pipeline* p, char* prompt, GenerationConfig* gen_config, char* output) {
    const GenerationConfig & config = *gen_config;
    auto streamer = std::make_shared<CallbackStreamer>(p, p->tokenizer.get());
    std::string result = p->generate(prompt, config, streamer.get());
    if (output != NULL) {
        std::strcpy(output, result.c_str());
    }
}

void CallbackStreamer::put(const std::vector<int> &output_ids) {
    if (is_prompt_) {
        // skip prompt
        is_prompt_ = false;
        return;
    }

    static const std::vector<char> puncts{',', '!', ':', ';', '?'};

    token_cache_.insert(token_cache_.end(), output_ids.begin(), output_ids.end());
    std::string text = tokenizer_->decode(token_cache_);
    if (text.empty()) {
        return;
    }

    std::string printable_text;
    if (text.back() == '\n') {
        // flush the cache after newline
        printable_text = text.substr(print_len_);
        token_cache_.clear();
        print_len_ = 0;
    } else if (std::find(puncts.begin(), puncts.end(), text.back()) != puncts.end()) {
        // last symbol is a punctuation, hold on
    } else if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
        // ends with an incomplete token, hold on
    } else {
        printable_text = text.substr(print_len_);
        print_len_ = text.size();
    }

    streamCallback(pipeline_, const_cast<char*>(printable_text.c_str()), 0);
}

void CallbackStreamer::end() {
    std::string text = tokenizer_->decode(token_cache_);
    streamCallback(pipeline_, const_cast<char*>(text.substr(print_len_).c_str()), 1);
    is_prompt_ = true;
    token_cache_.clear();
    print_len_ = 0;
}
