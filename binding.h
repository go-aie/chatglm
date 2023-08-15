#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

// This function is implemented in Go.
extern void streamCallback(void*, char*, int);

typedef struct GenerationConfig GenerationConfig;
typedef struct Pipeline Pipeline;

GenerationConfig* NewGenerationConfig(int max_length, int max_context_length, bool do_sample,
                                      int top_k, float top_p, float temperature,
                                      float repetition_penalty, int num_threads);
void DeleteGenerationConfig(GenerationConfig* p);

Pipeline* NewPipeline(char* path);
void DeletePipeline(Pipeline* p);
void Pipeline_Generate(Pipeline* p, char* prompt, GenerationConfig* gen_config, char* output);

#ifdef __cplusplus
}
#endif
