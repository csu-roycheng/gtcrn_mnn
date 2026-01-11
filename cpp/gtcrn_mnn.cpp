#include "gtcrn_mnn.h"

void GTCRNMNN::export_wav(
	const std::string& Filename,
	const std::vector<float>& Data,
	unsigned SampleRate) {
	AudioFile<float>::AudioBuffer Buffer;
	Buffer.resize(1);

	Buffer[0] = Data;
	size_t BufSz = Data.size();

	AudioFile<float> File;
	File.setAudioBuffer(Buffer);
	File.setAudioBufferSize(1, (int)BufSz);
	File.setNumSamplesPerChannel((int)BufSz);
	File.setNumChannels(1);
	File.setBitDepth(16);
	File.setSampleRate(SAMEPLERATE);
	File.save(Filename, AudioFileFormat::Wave);
}


void GTCRNMNN::inference(std::string input_audio_path, std::string output_audio_path) {

	AudioFile<float> input_file;
	input_file.load(input_audio_path);
	int audio_len = input_file.getNumSamplesPerChannel();
	int chunk_num = audio_len / HOP_LEN;
	if (audio_len % HOP_LEN != 0) {
		chunk_num += 1;
	}

	// init tensor
	auto input_tensor = MNN::Express::_Input({ 257,2 }, MNN::Express::NCHW, halide_type_of<float>());
	auto conv_cache_tensor = MNN::Express::_Input({ 1,16,16,66 }, MNN::Express::NCHW, halide_type_of<float>());
	auto tra_cache_tensor = MNN::Express::_Input({ 6,1,1,16 }, MNN::Express::NCHW, halide_type_of<float>());
	auto inter_cache_tensor = MNN::Express::_Input({ 2,33,16 }, MNN::Express::NCHW, halide_type_of<float>());
	float* input_ptr = input_tensor->writeMap<float>();
	float* conv_cache_ptr = conv_cache_tensor->writeMap<float>();
	float* tra_cache_ptr = tra_cache_tensor->writeMap<float>();
	float* inter_cache_ptr = inter_cache_tensor->writeMap<float>();

	std::vector<float> enhanced_data;

	// FFT config
	std::vector<size_t> shape{ BLOCK_LEN };
	std::vector<size_t> shape_ifft{ HOP_LEN };
	std::vector<size_t> axes{ 0 };
	std::vector<ptrdiff_t> stridel{ sizeof(double) };
	std::vector<ptrdiff_t> strideo{ sizeof(cpx_type) };

	double mic_in[BLOCK_LEN];
	double fft_input[BLOCK_LEN] = { 0 };

	std::vector<cpx_type> fft_output(BLOCK_LEN);

	for (int i = 0; i < chunk_num; i++) {
		std::memmove(mic_buffer, mic_buffer + HOP_LEN, (BLOCK_LEN - HOP_LEN) * sizeof(float));

		for (int n = 0; n < HOP_LEN; n++) {
			if (i * HOP_LEN + n < audio_len) {
				mic_buffer[BLOCK_LEN - HOP_LEN + n] = input_file.samples[0][i * HOP_LEN + n];
			}
			else {
				mic_buffer[BLOCK_LEN - HOP_LEN + n] = 0.0; // 填充0
			}
		}

		for (int n = 0; n < BLOCK_LEN; n++) {
			mic_in[n] = mic_buffer[n] * m_windows[n];
		}

		pocketfft::r2c(shape, stridel, strideo, axes, pocketfft::FORWARD, mic_in, fft_output.data(), 1.0);
		for (int n = 0; n < FFT_OUT_SIZE; n++) {
			spec[n * 2] = fft_output[n].real();
			spec[n * 2 + 1] = fft_output[n].imag();
		}

		std::memcpy(input_ptr, spec, 257 * 2 * sizeof(float));
		std::memcpy(conv_cache_ptr, conv_cache, 16 * 16 * 66 * sizeof(float));
		std::memcpy(tra_cache_ptr, tra_cache, 6 * 16 * sizeof(float));
		std::memcpy(inter_cache_ptr, inter_cache, 2 * 33 * 16 * sizeof(float));

		std::vector<MNN::Express::VARP> outputs;
		auto start = std::chrono::steady_clock::now();
		outputs = module->onForward({ input_tensor, conv_cache_tensor, tra_cache_tensor, inter_cache_tensor });
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double, std::milli> ms = end - start;
		std::cout << "耗时 " << ms.count() << " ms" << ", RTF: " << ms.count() / 1000 / 0.016 << std::endl;

		float y[FFT_OUT_SIZE * 2] = { 0 };
		auto enh = outputs[0]->readMap<float>();
		std::memcpy(y, enh, FFT_OUT_SIZE * 2 * sizeof(float));
		auto conv_cache_out = outputs[1]->readMap<float>();
		std::memcpy(conv_cache, conv_cache_out, 16 * 16 * 66 * sizeof(float));
		auto tra_cache_out = outputs[2]->readMap<float>();
		std::memcpy(tra_cache, tra_cache_out, 6 * 16 * sizeof(float));
		auto inter_cache_out = outputs[3]->readMap<float>();
		std::memcpy(inter_cache, inter_cache_out, 2 * 33 * 16 * sizeof(float));

		for (int n = 0; n < FFT_OUT_SIZE; n++) {
			enhanced_spec[n] = cpx_type(y[2 * n], y[2 * n + 1]);
		}

		pocketfft::c2r(shape, strideo, stridel, axes, pocketfft::BACKWARD, enhanced_spec, fft_input, 1.0 / BLOCK_LEN);

		for (int n = 0; n < HOP_LEN; n++) {
			enhanced_data.emplace_back(out_wav_buffer[n] + (float)fft_input[n] * m_windows[n]);
		}

		for (int n = 0; n < HOP_LEN; n++) {
			out_wav_buffer[n] = (float)fft_input[n + HOP_LEN] * m_windows[n + HOP_LEN];
		}
	}
	export_wav(output_audio_path, enhanced_data, SAMEPLERATE);
}
