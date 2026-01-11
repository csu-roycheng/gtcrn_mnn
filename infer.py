import MNN
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

class MNNInfer:

    def __init__(self, win_len=512, hop_len=256, model_path=None):
        
        self.sr = 16000
        self.win_len = win_len
        self.hop_len = hop_len
        self.win = np.hanning(512) ** 0.5

        config = {}
        config['backend'] = 0
        config['numThread'] = 1

        self.interpreter = MNN.Interpreter(model_path)
        self.session = self.interpreter.createSession(config)

        self.conv_cache = np.zeros([1, 16, 16, 66], dtype="float32")
        self.tra_cache = np.zeros([6, 1, 1, 16], dtype="float32")
        self.inter_cache = np.zeros([2, 33, 16], dtype="float32")
    
        self.input_cache = np.zeros([self.win_len,], dtype="float32")
        self.output_cache = np.zeros([self.hop_len,], dtype="float32")

        self.mix = np.zeros([257, 2], dtype="float32")
    

    def infer(self, wav_path):

        wav = librosa.load(wav_path, sr=self.sr, mono=True)[0]
        output = []
        
        input_tensor = self.interpreter.getSessionInput(self.session, "input")
        conv_cache_tensor = self.interpreter.getSessionInput(self.session, "conv_cache")
        tra_cache_tensor = self.interpreter.getSessionInput(self.session, "tra_cache")
        inter_cache_tensor = self.interpreter.getSessionInput(self.session, "inter_cache")
        yi_tensor = self.interpreter.getSessionOutput(self.session, "enh")
        conv_cache_output_tensor = self.interpreter.getSessionOutput(self.session, "conv_cache_out")
        tra_cache_output_tensor = self.interpreter.getSessionOutput(self.session, "tra_cache_out")
        inter_cache_output_tensor = self.interpreter.getSessionOutput(self.session, "inter_cache_out")

        for i in tqdm(range(0, wav.shape[-1], self.hop_len)):
            cur_frame = wav[i:i + self.hop_len]
            if len(cur_frame) < self.hop_len:
                cur_frame = np.concatenate([cur_frame, np.zeros([self.hop_len - len(cur_frame)], dtype="float32")], axis=-1)
            
            self.input_cache = np.concatenate([self.input_cache[self.hop_len:], cur_frame], axis=-1)
            
            spec = np.fft.rfft(self.input_cache * self.win).astype("complex64")
            self.mix[:, 0] = spec.real
            self.mix[:, 1] = spec.imag

            tmp_input = MNN.Tensor((257, 2), MNN.Halide_Type_Float, np.expand_dims(self.mix, axis=1), MNN.Tensor_DimensionType_Caffe)
            input_tensor.copyFrom(tmp_input)

            tmp_conv_cache = MNN.Tensor((1, 16, 16, 66), MNN.Halide_Type_Float, self.conv_cache, MNN.Tensor_DimensionType_Caffe)
            conv_cache_tensor.copyFrom(tmp_conv_cache)

            tmp_tra_cache = MNN.Tensor((6, 1, 1, 16), MNN.Halide_Type_Float, self.tra_cache, MNN.Tensor_DimensionType_Caffe)
            tra_cache_tensor.copyFrom(tmp_tra_cache)

            tmp_inter_cache = MNN.Tensor((2, 33, 16), MNN.Halide_Type_Float, self.inter_cache, MNN.Tensor_DimensionType_Caffe)
            inter_cache_tensor.copyFrom(tmp_inter_cache)

            self.interpreter.runSession(self.session)

            tmp_yi_output = MNN.Tensor((1, 257, 1, 2), MNN.Halide_Type_Float, np.ones([1, 257, 1, 2]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
            yi_tensor.copyToHostTensor(tmp_yi_output)
            yi = tmp_yi_output.getNumpyData()

            tmp_conv_cache_output = MNN.Tensor((1, 16, 16, 66), MNN.Halide_Type_Float, np.ones([1, 16, 16, 66]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
            conv_cache_output_tensor.copyToHostTensor(tmp_conv_cache_output)
            self.conv_cache = tmp_conv_cache_output.getNumpyData()

            tmp_tra_cache_output = MNN.Tensor((6, 1, 1, 16), MNN.Halide_Type_Float, np.ones([6, 1, 1, 16]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
            tra_cache_output_tensor.copyToHostTensor(tmp_tra_cache_output)
            self.tra_cache = tmp_tra_cache_output.getNumpyData()

            tmp_inter_cache_output = MNN.Tensor((2, 33, 16), MNN.Halide_Type_Float, np.ones([2, 33, 16]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
            inter_cache_output_tensor.copyToHostTensor(tmp_inter_cache_output)
            self.inter_cache = tmp_inter_cache_output.getNumpyData()
            
            spec_out = yi[0, :, :, 0] + 1j * yi[0, :, :, 1]
            enhanced = np.fft.irfft(spec_out.reshape(1, -1)) * self.win
            y = self.output_cache + enhanced[0, :self.hop_len]
            self.output_cache = enhanced[0, self.hop_len:]
            output.append(y)
        
        output = np.concatenate(output, axis=-1)
        sf.write("enh_mnn.wav", output.T, 16000)


if __name__ == "__main__":

    model = MNNInfer(model_path="gtcrn_stream.mnn")
    model.infer("mix.wav")