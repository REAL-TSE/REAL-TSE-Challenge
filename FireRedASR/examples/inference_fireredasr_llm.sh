#!/bin/bash

export PATH=$PWD/fireredasr/:$PWD/fireredasr/utils/:$PATH
export PYTHONPATH=$PWD/:$PYTHONPATH

# model_dir includes model.pth.tar, asr_encoder.pth.tar, cmvn.ark, Qwen2-7B-Instruct
model_dir=$PWD/pretrained_models/FireRedASR-LLM-L

# Support several input format
wavs="--wav_path wav/BAC009S0764W0121.wav"
wavs="--wav_paths wav/BAC009S0764W0121.wav wav/IT0011W0001.wav wav/TEST_NET_Y0000000000_-KTKHdZ2fb8_S00000.wav wav/TEST_MEETING_T0000000001_S00000.wav"
wavs="--wav_dir wav/"
wavs="--wav_scp wav/wav.scp"

out="out/llm-l-asr.txt"

decode_args="
--batch_size 1 --beam_size 3 --decode_max_len 0 --decode_min_len 0
--repetition_penalty 3.0 --llm_length_penalty 1.0 --temperature 1.0
"

mkdir -p $(dirname $out)
set -x


CUDA_VISIBLE_DEVICES=0 \
speech2text.py --asr_type "llm" --model_dir $model_dir $decode_args $wavs --output $out


ref="wav/text"
wer.py --print_sentence_wer 1 --do_tn 0 --rm_special 1 --ref $ref --hyp $out > $out.wer 2>&1
tail -n8 $out.wer
