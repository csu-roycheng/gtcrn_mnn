#include "pocketfft_hdronly.h"
#include "AudioFile.h"
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <iostream>
#include <chrono>

#define PI 3.14159265358979323846
#define SAMEPLERATE  (16000)
#define BLOCK_LEN		(512)
#define HOP_LEN  (256)
#define FFT_OUT_SIZE (257)
typedef std::complex<double> cpx_type;


class GTCRNMNN {
public:
    void inference(std::string input_audio_path, std::string output_audio_path);

    GTCRNMNN(const char* ModelPath) {
        // Init MNN model
        init_mnn_model(ModelPath);
        for (int i = 0; i < BLOCK_LEN; i++) {
            m_windows[i] = sinf(PI * i / (BLOCK_LEN - 1));
        }
    };

private:
    // MNN resources
    MNN::BackendConfig backend_config;
    std::shared_ptr<MNN::Express::Executor> executor;
    std::unique_ptr<MNN::Express::Module> module;

    // cache settings
    double m_windows[BLOCK_LEN] = { 0 };

    float mic_buffer[BLOCK_LEN] = { 0 };
	float out_wav_buffer[HOP_LEN] = { 0 };

    float conv_cache[16 * 16 * 66] = { 0 };
    float tra_cache[6 * 16] = { 0 };
    float inter_cache[2 * 33 * 16] = { 0 };

    float spec[FFT_OUT_SIZE * 2] = { 0 };
    cpx_type enhanced_spec[FFT_OUT_SIZE] = { 0 };

    std::vector<std::string> input_node_names{ "input", "conv_cache", "tra_cache", "inter_cache" };
    std::vector<std::string> output_node_names{ "enh", "conv_cache_out", "tra_cache_out", "inter_cache_out" };

    void export_wav(const std::string& Filename, const std::vector<float>& Data, unsigned SampleRate);

    void init_mnn_model(const char* ModelPath) {
        // 创建 Executor
        executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backend_config, 1);
        // 绑定Executor，在创建/销毁/使用Module或进行表达式计算之前都需要绑定
        MNN::Express::ExecutorScope _s(executor);

        //创建 Module
        module.reset(MNN::Express::Module::load(input_node_names, output_node_names, ModelPath));
    };
};
